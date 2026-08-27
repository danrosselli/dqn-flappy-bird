---
layout: layouts/insight.njk
title: "Q-values and action selection — how the network decides"
description: The network outputs two numbers per state; epsilon-greedy picks which one to follow. A walkthrough of Q-values, argMax, and the exploration-exploitation switch.
section: insights
tags: insight
category: architecture
order: 8
updated: 2026-08-27
keywords: ["Q-values", "action selection", "argMax", "epsilon greedy", "choose action", "neural network output"]
keyIdea: "The network doesn't decide 'flap or not' — it estimates two numbers (Q_IDLE, Q_FLAP), and a separate function picks the action via epsilon-greedy."
relatedExperiments: []
references: []
---

## What the network outputs

The final layer has **2 neurons** with linear activation:

```javascript
model.add(tf.layers.dense({ units: 2, activation: 'linear' }));
```

For any state, the network produces a vector of two Q-values:

```text
[Q(state, IDLE),  Q(state, FLAP)]
     index 0           index 1
```

These numbers represent the estimated discounted future reward for each action in
that state. They start out nearly random and improve as training progresses.

## How chooseAction works

```javascript
async chooseAction(state) {
  // 1. Exploration branch
  if (Math.random() < epsilon) {
    return Math.random() < 0.05 ? ACTION_FLAP : ACTION_IDLE;
  }

  // 2. Exploitation branch
  return tf.tidy(() => {
    const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
    const qValues = this.model.predict(stateTensor);
    const action = qValues.argMax(1).dataSync()[0];
    return action;
  });
}
```

Two branches, one decision:

| Condition | Behavior | Outcome |
|---|---|---|
| `Math.random() < epsilon` | Random with 5% flap bias | 0 (IDLE) or 1 (FLAP) |
| Otherwise | Pick the action with highest Q | 0 or 1 |

The **5% flap bias** in the exploration branch is a domain-specific choice: uniform
random flapping kills the bird almost immediately, so exploration is biased toward
gliding (IDLE) to produce longer, more informative trajectories.

## argMax and dataSync

Two TensorFlow.js methods make the exploitation branch work:

**`argMax(1)`** — returns the *index* of the largest value along axis 1 (the action
axis). For a tensor of shape `[1, 2]`, it picks whichever of Q_IDLE or Q_FLAP is
larger.

**`dataSync()`** — copies the result from GPU/WebGL memory to a CPU TypedArray.
Synchronous and blocking, but fine for a single scalar.

```javascript
const action = qValues.argMax(1).dataSync()[0];
//  argMax(1)    → tensor with index of best action
//  .dataSync()  → TypedArray [0] or [1]
//  [0]          → the number itself
```

## Why the network doesn't decide directly

A common misconception: the network "tells the bird what to do." In reality, the
network is a function approximator for Q(s, a) — it assigns a value to each
possible action. The **decision logic** lives in `chooseAction`, which compares
those values and adds the exploration noise. Separating estimation from decision
keeps the architecture clean and makes it easy to swap exploration strategies
without retraining.
