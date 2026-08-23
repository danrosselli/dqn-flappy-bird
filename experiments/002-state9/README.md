# Absolute bird position state (9D)

## Objective

Test whether adding the bird's absolute vertical position on screen (`birdY`) to the state vector improves control quality over the purely pipe-relative 8-dimensional state of the baseline.

## Hypothesis

The baseline state describes the bird only through relative distances to pipes and its own velocity — it never tells the agent *where on the screen* it is. An absolute position feature should give direct awareness of vertical bounds and room to maneuver, improving flap timing near the ceiling and floor.

## Changes

- Added `birdY` state feature: `(bird.y / scale.height) * 2 - 1`, range [-1, +1].
- `STATE_SIZE` increased from 8 to 9; first dense layer input shape updated accordingly.
- Everything else (hyperparameters, rewards, buffer, exploration) identical to Experiment 001.

## Training

Episodes: 2560
Learning rate: 0.001
Gamma: 0.99
Batch size: 64
Target update interval: 1000
Train throttle: 2
Sampling strategy: sampleRandomBasic
Epsilon: 0.9 → 0.0 (decay 0.9995)

## Results

Best score: 456
Episodes: 2560

Baseline reference (Experiment 001): best score 485 in 2160 episodes.

## Conclusion

No improvement was observed: the result sits slightly below baseline, within what run-to-run noise would plausibly explain. The relative pipe features appear to carry all the signal needed for this task; absolute screen position did not add useful information under otherwise identical training conditions. The feature remains available for future experiments where absolute positioning may matter more.
