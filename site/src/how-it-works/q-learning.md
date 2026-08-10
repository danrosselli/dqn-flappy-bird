---
layout: layouts/article.njk
title: Q-Learning
description: The learning rule at the core of the agent — Q-values, the Bellman equation, and how bootstrapped targets are built.
section: how-it-works
sectionTitle: How It Works
collection: howItWorks
tags: how-it-works
order: 6
---

Q-learning learns a function **Q(s, a)**: *"how good is action `a` in state `s`,
measured as discounted future reward?"* If that function is accurate, acting well is
trivial — always pick the action with the highest Q.

## The Bellman equation

The definition of the optimal Q-function is recursive:

```text
Q*(s, a) = r + γ · max Q*(s′, a′)
                  a′
```

The value of an action now equals the immediate reward plus the discounted value of
whatever the best follow-up looks like. With γ = **0.99**, the agent weighs rewards
roughly a hundred steps into the future — long enough to connect "flap now" with
"cross the pipe two seconds later".

## From equation to gradient descent

The network can't store a table, so it regresses toward the Bellman target. For each
transition in a batch:

```text
target = r + γ · max Q_target(s′) · (1 − done)
loss   = MSE( Q_online(s, a), target )
```

Only the Q-value of the action that was actually taken is updated; the other output is
left at its prediction. Note the two networks:

- **`Q_online`** — the network being trained, updated every optimization step.
- **`Q_target`** — a frozen copy used only to *compute targets*, re-synced from the
  online network every **1,000 steps**.

Without the target network, the same moving network would produce both predictions and
targets — a feedback loop that tends to oscillate or diverge. The frozen copy makes
each training round a stable supervised problem.

## Terminal transitions

When `done` is true, the bootstrap term is masked out and the target collapses to
`r` alone. This is how the "game over" signal propagates: states just before death
get pulled toward −20, states before those get pulled toward something slightly less
bad, and so on backward through the trajectory.

## Why it works here

Q-learning is **off-policy**: it learns about the greedy policy even while behaving
ε-greedily. That pairs perfectly with experience replay — the agent can store whatever
it experienced, sample it in any order, and still learn the value of acting optimally.
The full training mechanics live in [Training]({{ '/training/' | url }}).
