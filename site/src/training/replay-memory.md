---
layout: layouts/article.njk
title: Replay Memory
description: Experience replay — a 50k-transition memory with reservoir and recency structures, and the samplers kept for experimentation.
section: training
sectionTitle: Training
collection: training
tags: training
order: 3
---

Online Q-learning from a raw frame stream learns from a firehose of nearly identical
frames: highly correlated, and quickly forgotten. **Experience replay** fixes both
problems — store transitions, then train on random samples of the past.

## What a transition contains

```text
( state[8], action ∈ {0,1}, reward, nextState[8], done )
```

One frame of experience, complete enough to compute a Bellman target. Frames with no
pipe on screen (`dx ≥ 1`) are refused at the gate — they carry no obstacle signal.

## The buffer is three structures

The `ReplayBuffer` class maintains more than one view of the past:

| Structure | Size | Role |
| --- | --- | --- |
| Legacy circular buffer | 50,000 | The training source — uniform random sampling |
| Recent buffer | 10,000 | The freshest transitions, circularly overwritten |
| Reservoir | 40,000 | Long-term memory with uniform survival probability |

The **reservoir** implements classic reservoir sampling: once full, each new
transition replaces a random slot with probability `size / totalSeen`, so old and new
experiences coexist with no recency bias. A floor keeps the replacement chance at
**≥ 5%** even after millions of frames.

## Sampling strategies

Training currently samples **uniformly from the legacy buffer** (`sampleRandomBasic`).
The class deliberately keeps alternative samplers in the codebase — each one a
hypothesis waiting for an experiment:

- **Recency-weighted** — a fixed share of the batch from the newest transitions.
- **Reward-prioritized** — sampling proportional to |reward|, so rare −20 and +10
  events appear more often than +0.1 filler.
- **Hybrid recent + reservoir** — ~35% recent, the rest long-term.
- **Sequential-recent hybrid** — always includes the newest frames, chronologically.
- **Crash-focus** — when a recent death is found, a share of the batch comes from the
  trajectory that led into it.

Which sampler helps, hurts, or does nothing is exactly the kind of question the
[Experiments]({{ '/experiments/' | url }}) section exists to answer with data.
