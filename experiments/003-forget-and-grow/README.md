# Forget-and-Grow (FoG)

## Objective

Evaluate a Forget-and-Grow strategy against two known failure modes of long-running DQN: **primacy bias** (overfitting to early, easier experiences) and **loss of plasticity** (dormant units unable to adapt as the environment shifts). Flappy Bird is continuously non-stationary — pipe speed rises with score — so both effects are expected to appear.

## Hypothesis

Following Kang et al. (2025), *A Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control* (arXiv:2507.02712): decaying the influence of old experience (ER decay) and periodically adding fresh network capacity (network expansion) should improve stability and long-run performance over a fixed-size baseline.

## Strategy

FoG combines two mechanisms:

1. **ER Decay** — sampling weights for the reservoir decay with transition age (`decayFactor` 0.99992, floor 0.05), implemented as `sampleHybridWithDecay`. **Not active in this run**: training deliberately sampled with `sampleRandomBasic` to isolate the effect of network expansion.
2. **Network Expansion** — every 80 generations (`expandEveryGenerations`), both hidden layers grow by 16 units (`expandUnits`). Old weights are preserved and copied into their original positions; new units are randomly initialized. Growth is capped at 192 hidden units (`maxHiddenUnits`). The target network is recreated from the expanded model, and epsilon receives a small boost (+0.05) after each expansion to encourage exploration with the new capacity.

The full specification lives in [README_Forget_and_Grow.md](./README_Forget_and_Grow.md) (Portuguese).

## Changes

- Weight-preserving `expandNetwork()` / `maybeExpandNetwork()`, triggered on death every 80 generations.
- ER-decay sampling method added to the replay buffer (available but inactive in this run).
- Persistence extended: hidden unit count and expansion count saved/restored with the brain.
- Epsilon boost after each expansion.

## Training

Episodes: 2265
Learning rate: 0.001
Gamma: 0.99
Batch size: 64
Target update interval: 1000
Train throttle: 2
Sampling strategy: sampleRandomBasic
Initial hidden units: 64
Expand every generations: 80
Expand units: 16
Max hidden units: 192
Epsilon boost on expand: 0.05

## Results

Best score: 202
Episodes: 2265

Baseline reference (Experiment 001): best score 485 in 2160 episodes.

## Conclusion

Performance dropped well below baseline. Within 2265 episodes the network expanded up to ~28 times, repeatedly re-randomizing part of the policy's parameters while the environment kept accelerating; the observed degradation is consistent with expansions disrupting learned behavior faster than the new capacity could pay off. Note that this run tested expansion **in isolation** — full FoG (with ER decay active) was not evaluated. Before discarding the approach, follow-up experiments should either enable ER decay or reduce expansion frequency.
