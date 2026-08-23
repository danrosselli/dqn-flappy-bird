# Baseline DQN

## Objective

Establish the project's baseline: a standard DQN implementation assembled from well-established techniques in the deep RL literature — experience replay, target network, and epsilon-greedy exploration — configured to learn Flappy Bird from low-dimensional state inputs.

## Hypothesis

A compact MLP fed with relative pipe geometry, plus shaped rewards, should be sufficient to learn a competent flapping policy and produce a strong reference result for all subsequent experiments.

## Approach

- **State (8-dim)**: horizontal distance and gap alignment for the current and next pipe (`dx`, `dy`, `gap`, `dxNext`, `dyNext`, `gapNext`), bird vertical velocity (`velY`) and current pipe speed (`speed`), all normalized to [-1, +1].
- **Rewards**: survival bonus (+0.05/frame), pipe passage (+10), collision (-20), velocity penalty (-0.05 when |velY| > 700), and Gaussian proximity shaping toward the current gap center.
- **Replay**: hybrid buffer combining a circular recent buffer (10k), reservoir sampling for long-term memory (40k), and a legacy circular buffer (50k). States with no pipe visible (dx >= 1) are discarded.
- **Training**: target network hard-updated every 1000 steps, batch 64, Adam lr 0.001, gamma 0.99, train throttle 2.
- **Exploration**: epsilon-greedy from 0.9 decaying by 0.9995 per successful update down to 0. Note that epsilon only decays when a batch actually trains, so early exploration lasts longer than a naive per-episode estimate suggests.

This configuration follows the default recipe found across prior DQN studies, adapted to the game's non-stationarity (pipe speed rises with score).

## Changes

- Initial implementation: `DQNAgent` on TensorFlow.js with sequential 8→64→64→2 network.
- Hybrid reservoir replay buffer with six interchangeable sampling strategies.
- Shaped reward function with proximity term.
- Full persistence (weights, replay buffers, epsilon, generation counter) via IndexedDB.
- In-game HUD with score, generation, epsilon, raw state values and Q-values.

## Training

Episodes: 2160
Learning rate: 0.001
Gamma: 0.99
Batch size: 64
Target update interval: 1000
Train throttle: 2
Sampling strategy: sampleRandomBasic
Epsilon: 0.9 → 0.0 (decay 0.9995)

## Results

Best score: 485
Episodes: 2160

## Conclusion

The baseline fulfilled its purpose: a standard DQN recipe was sufficient to reach a good result at the time, without any project-specific innovation. Experiment 001 is the reference point that every later experiment compares against.
