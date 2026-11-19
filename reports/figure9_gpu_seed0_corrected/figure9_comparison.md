# HalfCheetah comparison with Figure 9

The run reached **320,000 episodes** on **NVIDIA GeForce RTX 4060 Ti** in **451.44 minutes** (7.52 hours). The measured bounds are Lmin=8 and plotted upper bound=21.

Figure 9 runs to about **320,000 episodes**. Linear extrapolation from this run gives **7.52 hours** (0.31 days) from scratch, or **0.00 additional hours** from this checkpoint. This assumes constant throughput. A short pilot may not capture the added overhead of adaptive evaluations and checkpoints.

The reference transitions below are approximate readings from the paper's plot. The paper does not report exact transition episodes. The plot uses its upper bound labels directly and interprets the run's stored inclusive high value as high + 1.

| Episode | Run Lmin | Figure 9 Lmin | Run upper | Figure 9 upper |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 10 | 11 | 11 |
| 50,000 | 8 | 10 | 11 | 11 |
| 65,000 | 8 | 8 | 11 | 13 |
| 70,000 | 8 | 7 | 11 | 13 |
| 100,000 | 8 | 7 | 11 | 14 |
| 205,000 | 8 | 7 | 15 | 14 |
| 230,000 | 8 | 6 | 18 | 15 |
| 240,000 | 8 | 6 | 18 | 16 |
| 255,000 | 8 | 5 | 18 | 16 |
| 320,000 | 8 | 5 | 21 | 16 |

At episode 320,000, the paper trace is approximately Lmin=5 and upper=16. Run-minus-reference differences are +3 and +5.

This run covers the full plotted episode horizon.
