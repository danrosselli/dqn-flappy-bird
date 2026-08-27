---
layout: layouts/insight.njk
title: "The Bellman equation — teaching the network what to predict"
description: How the Bellman target is computed in this DQN, why the max over future actions makes it off-policy, and where the equation lives in the code.
section: insights
tags: insight
category: training
order: 6
updated: 2026-08-27
keywords: ["bellman equation", "temporal difference", "off-policy", "target computation", "bootstrapping"]
keyIdea: "The Bellman equation defines what the network should have predicted — the immediate reward plus discounted future value — and the max over future actions is what makes DQN off-policy."
relatedExperiments: []
references:
  - authors: "R. S. Sutton, A. G. Barto"
    title: "Reinforcement Learning: An Introduction (2nd ed.)"
    venue: "MIT Press"
    year: 2018
    url: "http://incompleteideas.net/book/the-book-2nd.html"
    label: "incompleteideas.net/book/the-book-2nd.html"
  - authors: "V. Mnih, K. Kavukcuoglu, D. Silver, et al."
    title: "Human-level control through deep reinforcement learning"
    venue: "Nature, 518"
    year: 2015
    url: "https://doi.org/10.1038/nature14236"
    label: "doi:10.1038/nature14236"
---

## What the equation says

The optimal Q-function satisfies a recursive identity:

```text
Q*(s, a) = r + γ · max_a′ Q*(s′, a′)
```

The value of taking action `a` in state `s` equals the immediate reward plus the
discounted value of the best thing you can do next. With γ = **0.99**, the agent
connects "flap now" with outcomes roughly a hundred frames into the future.

## Where it lives in the code

The equation appears **exclusively** in the `train()` method of `DQNAgent`
(`src/rl/dqn.js`). The game code never computes Bellman targets — it only
observes states, calculates rewards, stores transitions, and calls `train()`.

The core lines:

```javascript
const nextQValues = this.targetModel.predict(nextStateTensor);
const maxNextQ = nextQValues.max(1);
const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));
```

Each piece maps directly to the math:

| Code | Math | Meaning |
|---|---|---|
| `rewardTensor` | r | Immediate reward |
| `maxNextQ` | max_a′ Q(s′, a′) | Best future value from target network |
| `gamma` | γ = 0.99 | Discount factor |
| `notDone` | (1 − done) | Zeros the future at episode end |

## Why the max matters: off-policy vs on-policy

The max over future actions is a deliberate design choice with a clear consequence:

```text
Q-Learning / DQN:  Q(s,a) ← r + γ · max_a′ Q(s′, a′)   → off-policy
SARSA:             Q(s,a) ← r + γ · Q(s′, a′)            → on-policy
```

DQN uses the **max** because it wants to estimate Q\* — the value under the
*optimal* future policy — regardless of what the agent actually did in `s′`. This
is essential here because the ε-greedy exploration policy often takes suboptimal
actions. Using the actual action taken would contaminate the target with
exploration noise.

## Terminal transitions

When `done` is true (the bird died), the `notDone` mask zeros out the bootstrap
term and the target collapses to `r` alone. This is how the death signal propagates
backward: the frame before death gets pulled toward −20, the frame before that
toward something slightly less bad, and so on through the trajectory.

## What the network actually learns

The target is used to update **only the Q-value of the action that was taken**:

```text
qValuesArray[i][actions[i]] = target;
```

The other output (the action not taken) keeps its current prediction and contributes
near-zero loss. Over many batches, both outputs converge — but each gradient step
only pushes one of them toward the Bellman target.
