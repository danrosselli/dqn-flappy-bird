---
layout: layouts/insight.njk
title: Experience replay — learning from the past without drowning in it
description: Why replay memory exists, what this project's three-buffer structure buys, and where prioritization might matter.
section: insights
tags: insight
category: replay
order: 3
updated: 2026-08-10
keywords: ["experience replay", "replay buffer", "sample efficiency", "prioritized replay"]
keyIdea: "Replay decorrelates training data and multiplies the value of rare events — but 'which past deserves to be re-sampled' is its own research question, which is why this codebase keeps six samplers alive."
relatedExperiments: []
references:
  - authors: "L.-J. Lin"
    title: "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching"
    venue: "Machine Learning, 8"
    year: 1992
    url: "https://link.springer.com/article/10.1007/BF00992699"
    label: "doi:10.1007/BF00992699"
  - authors: "T. Schaul, J. Quan, I. Antonoglou, D. Silver"
    title: "Prioritized Experience Replay"
    venue: "International Conference on Learning Representations (ICLR)"
    year: 2016
    url: "https://arxiv.org/abs/1511.05952"
    label: "arxiv.org/abs/1511.05952"
---

## Why replay at all

Lin introduced experience replay in 1992 for reasons that are still the reasons:
consecutive frames are nearly identical, so training on them in order violates the
i.i.d. assumptions that make stochastic gradient descent behave, and each experience
is used once and discarded — a terrible deal when the interesting ones are rare.

In this project the interesting events really are rare: +10 pipe crossings and −20
deaths are drops in a sea of +0.1 survivals. Replay gives those drops hundreds of
chances to matter.

## The three-buffer structure

The `ReplayBuffer` keeps a 50k circular buffer (the training source), a 10k recent
buffer (what just happened), and a 40k reservoir (a uniform long-term memory with a
guaranteed minimum replacement chance). The design tension being managed:

- **Recency** matters because the policy is changing — old transitions were produced
  by a worse agent.
- **Long memory** matters because competence at speed 200 shouldn't be forgotten by
  the time the world reaches 400.

## What each transition carries

Every transition stored in the buffer is a five-tuple:

| Field | Meaning | Example in Flappy Bird |
|---|---|---|
| `state` | The 8-value vector before the action | `[dx, dy, velY, gap, dxNext, ...]` |
| `action` | What the agent chose (0 = IDLE, 1 = FLAP) | `0` or `1` |
| `reward` | The scalar reward for that step | survival + proximity + bonuses |
| `nextState` | The 8-value vector after the action | new `[dx, dy, velY, ...]` |
| `done` | Whether the episode ended | `true` / `false` |

The buffer's `add()` method (`src/rl/replayBuffer.js`) appends these into the
circular storage, overwriting the oldest entries when capacity is reached.

## From buffer to training

Training starts by sampling a random mini-batch:

```javascript
const batch = this.replayBuffer.sampleRandomBasic(BATCH_SIZE);  // BATCH_SIZE = 64
```

The 64 transitions are split into parallel arrays (`states`, `actions`, `rewards`,
`nextStates`, `dones`) and converted to tensors. Then the Bellman target is
computed for each:

```text
target = reward + γ · max Q_target(nextState) · (1 − done)
```

The online network's prediction for the taken action is compared against this
target, and only that output is updated via `model.fit()`.

## The TD-error quality gate

Before committing to a `model.fit`, the code checks whether the batch is worth
training on:

```text
For each transition in the batch:
  normError = |target − predicted_Q| / (|target| + ε)

meanNormTDError = average(normErrors)

if meanNormTDError >= TD_NORM_THRESHOLD (0.04):
    train the batch
else:
    skip — the network already predicts well enough
```

This is a batch-level heuristic: if the average normalized error is small, training
would add noise rather than signal. It's cheaper than per-sample prioritization
and avoids wasting compute on batches the network already understands.

## Where prioritization could enter

Schaul et al.'s prioritized replay samples transitions by TD-error magnitude — learn
from what surprises you. The codebase already contains a reward-magnitude variant and
a crash-focus variant as study artifacts. Whether either beats uniform sampling here
is an empirical question, and a clean candidate for an early numbered experiment:
identical everything, different sampler, comparable scores.
