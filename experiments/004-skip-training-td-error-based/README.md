# Experiment 004 — TD-error-gated training

## Objective

Evaluate whether gating gradient updates on the batch TD-error improves training efficiency, by skipping updates that carry little learning signal instead of running them unconditionally on every sampled batch.

## Hypothesis

In later stages of training most sampled batches are already well predicted by the network, so their gradient contributions are near zero while still consuming compute and decaying epsilon. If updates run only for batches whose mean normalized TD-error exceeds a threshold, the agent should reach high scores with far fewer training steps.

## Strategy

Each `train()` call samples a batch (`sampleRandomBasic`) and computes, before fitting:

- the raw TD-error per transition: `|target − Q(s,a)|`;
- the normalized TD-error: `|target − Q(s,a)| / (|target| + 1e-6)`.

The gradient update runs only when the **mean normalized TD-error of the batch** is at or above `tdNormThreshold` (0.04); otherwise the whole batch is skipped and nothing is backpropagated.

Normalizing by `|target|` matters because Q-values grow as returns accumulate: an absolute error of 0.5 is large early in training but negligible once Q-values reach the hundreds. The ratio keeps a single fixed threshold meaningful across the entire training curve — which is why this experiment replaced the earlier adaptive history-percentile threshold design.

One side effect shapes exploration: epsilon decays only on batches that actually train (`decayEpsilon()` sits inside the trained branch). Long stretches of skipped batches therefore also pause exploration decay — skipping compute buys slower epsilon decay for free. This means the final epsilon depends on how many updates actually happened, not on how many batches were sampled; the "Final ε" estimate shown on the website assumes one decay per episode and is optimistic for this experiment.

## Changes

- Computed raw and normalized batch TD-error inside the target calculation.
- Gated `model.fit()` on `meanNormTDError >= TD_NORM_THRESHOLD` (0.04).
- Skipped batches no longer decay epsilon.
- HUD now shows training statistics: trained/attempts ratio, last normalized TD-error, and current threshold.

## Training

Episodes: 290
Learning rate: 0.001
Gamma: 0.99
Batch size: 64
Target update interval: 1000
Train throttle: 2
Sampling strategy: sampleRandomBasic
TD norm threshold: 0.04

## Results

Best score: 5540
Episodes: 290

For reference, the baseline (Experiment 001) reached best score 485 in 2160 episodes under identical core hyperparameters, without update gating.

## Conclusion

TODO — the single-run comparison above (11× the baseline best score with ~7× fewer episodes) is consistent with the hypothesis, but more runs are needed before drawing a firm conclusion.
