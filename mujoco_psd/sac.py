"""Continuous-action SAC used by the state-based MuJoCo PSD runner.

The network widths, discount, Polyak coefficient, and base learning rate follow
the PSD paper's Appendix C.2/C.6 settings. The squashed Gaussian actor,
entropy-temperature tuning, twin critics, and target critics follow the
referenced SAC implementation pattern.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


HIDDEN_DIM = 1024
GAMMA = 0.99
TAU = 0.005  # Polyak averaging coefficient corresponding to target smoothing 0.995.
LEARNING_RATE = 1e-4
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, output_dim),
    )


class _GaussianActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, action_scale: torch.Tensor,
                 action_bias: torch.Tensor):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, action_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def sample(self, observation: torch.Tensor, deterministic: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.trunk(observation)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mean) * self.action_scale + self.action_bias, None

        # The scale is positive by construction. Validate losses and actions
        # in the training loop without synchronizing every GPU distribution.
        distribution = Normal(mean, log_std.exp(), validate_args=False)
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        # Change of variables for tanh and the affine action-space transform.
        log_probability = distribution.log_prob(pre_tanh)
        log_probability -= torch.log(1.0 - squashed.square() + 1e-6)
        log_probability -= torch.log(self.action_scale)
        return action, log_probability.sum(dim=-1, keepdim=True)


class _Critic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.q1 = _mlp(input_dim + action_dim, 1)
        self.q2 = _mlp(input_dim + action_dim, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat((observation, action), dim=-1)
        return self.q1(value), self.q2(value)


class SAC:
    """Twin-critic SAC for a Gym-compatible continuous action space.

    ``action_space`` is only inspected for ``low`` and ``high`` arrays, so this
    module does not import Gymnasium or depend on a specific environment class.
    Observations may be passed as ``(obs_dim,)`` or ``(batch, obs_dim)`` to
    :meth:`act`. ``terminal`` in :meth:`update` must distinguish true terminal
    states from time-limit truncations. The caller should pass zero for a
    time-limit truncation so that the value target bootstraps across it.
    """

    def __init__(self, input_dim: int, action_space: Any, device: torch.device):
        self.device = torch.device(device)
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device).reshape(-1)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device).reshape(-1)
        if low.numel() != high.numel() or not torch.all(torch.isfinite(low)) or not torch.all(torch.isfinite(high)):
            raise ValueError("action_space low and high must be finite arrays of equal size")
        if not torch.all(high > low):
            raise ValueError("every action_space upper bound must exceed its lower bound")

        self.action_dim = int(low.numel())
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0
        self.actor = _GaussianActor(input_dim, self.action_dim, action_scale, action_bias).to(self.device)
        self.critic = _Critic(input_dim, self.action_dim).to(self.device)
        # The SAC implementation cited by PSD initializes linear layers this way.
        for network in (self.actor, self.critic):
            for layer in network.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.target_critic = _Critic(input_dim, self.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        fused = self.device.type == "cuda"
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE, fused=fused)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE, fused=fused)
        self.log_alpha = torch.zeros((), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        # Standard SAC automatic target entropy for a continuous action vector.
        self.target_entropy = -float(self.action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        observation = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        unbatched = observation.ndim == 1
        if unbatched:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2:
            raise ValueError("observations must have shape (obs_dim,) or (batch, obs_dim)")
        with torch.no_grad():
            action, _ = self.actor.sample(observation, deterministic=deterministic)
        return action.squeeze(0) if unbatched else action

    def update(self, x: torch.Tensor, action: torch.Tensor, next_x: torch.Tensor,
               reward: torch.Tensor, terminal: torch.Tensor | None) -> dict[str, torch.Tensor]:
        observation = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        next_observation = torch.as_tensor(next_x, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).reshape(-1, 1)
        if observation.ndim != 2 or next_observation.shape != observation.shape:
            raise ValueError("x and next_x must have matching shape (batch, obs_dim)")
        if action.shape != (observation.shape[0], self.action_dim):
            raise ValueError("action must have shape (batch, action_dim)")
        if reward.shape[0] != observation.shape[0]:
            raise ValueError("reward batch size must match x")
        if terminal is not None:
            terminal = torch.as_tensor(terminal, dtype=torch.float32, device=self.device).reshape(-1, 1)
            if terminal.shape[0] != observation.shape[0]:
                raise ValueError("terminal batch size must match x")
            if torch.any((terminal < 0) | (terminal > 1)):
                raise ValueError("terminal entries must be in [0, 1]")

        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(next_observation)
            target_q1, target_q2 = self.target_critic(next_observation, next_action)
            target_value = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_probability
            if terminal is not None:
                target_value = (1.0 - terminal) * target_value
            target = reward + GAMMA * target_value

        current_q1, current_q2 = self.critic(observation, action)
        critic_loss = nn.functional.mse_loss(current_q1, target) + nn.functional.mse_loss(current_q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_action, log_probability = self.actor.sample(observation)
        self.critic.requires_grad_(False)
        policy_q1, policy_q2 = self.critic(observation, sampled_action)
        actor_loss = (self.alpha.detach() * log_probability - torch.minimum(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        # The target differential entropy is negative for this action space.
        alpha_loss = -(self.log_alpha * (log_probability.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, TAU)

        return {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.log_alpha.detach().exp(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return model and optimizer state, including the learned temperature."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "target_entropy": self.target_entropy,
            "action_dim": self.action_dim,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError("checkpoint action dimension does not match action_space")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        self.target_entropy = float(state["target_entropy"])
