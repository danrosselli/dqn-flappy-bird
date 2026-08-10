---
layout: layouts/section.njk
title: The Project
description: What DQN Flappy Bird is, why it exists, and how it evolves — an ongoing reinforcement learning experiment documented as a public lab notebook.
section: project
---

## What is this?

**DQN Flappy Bird** is a personal research and learning experiment. A reinforcement
learning agent — a Deep Q-Network — learns to play Flappy Bird autonomously, and the
whole process happens **in the browser**: the game (Phaser 3), the neural network
(TensorFlow.js), the training loop, and the persistence layer (IndexedDB) all live
client-side, in a single tab.

There is no pre-trained model shipped with the project. Open the page and the agent
starts naive — flapping at random, dying immediately — and improves as episodes
accumulate. Close the tab, come back tomorrow, and it resumes where it left off.

{% include "components/loop-diagram.njk" %}

## Why Flappy Bird?

Flappy Bird is close to an ideal laboratory environment for reinforcement learning:

- **Simple rules, hard problem.** One input (flap or not), unforgiving consequences.
- **Continuous dynamics.** Gravity, momentum, and moving obstacles produce a smooth,
  physics-like state space rather than a grid.
- **Immediate, unambiguous feedback.** Passing a pipe is success; touching anything is
  death. Credit assignment is clean.
- **Episodic structure.** Each life is a self-contained episode, which maps naturally
  onto Q-learning.
- **It punishes naivety.** A random agent dies in seconds, so improvement is visible
  and measurable.

The environment also has a twist of its own: in this implementation, **pipe speed grows
with the score**. The better the agent gets, the faster the world moves — a built-in
difficulty curriculum.

## Why reinforcement learning?

The project is an excuse to study RL hands-on. Reinforcement learning is the right
frame for this problem because there is no dataset of "correct moves" — only a reward
signal. The agent has to discover, through trial and error, *when* flapping is a good
idea. That discovery process is the interesting part, and it is exactly what this site
documents.

## Why DQN?

Q-learning is the classical algorithm for learning action values, but its table-based
form cannot represent a continuous state space. **Deep Q-Learning** replaces the table
with a neural network that approximates Q(s, a), making it possible to feed the agent
raw, continuous measurements — distances, velocities, gap geometry — without manual
discretization.

DQN also brings the two stabilizing ideas that made it famous, both implemented here:

- **Experience replay** — transitions are stored in a 50k-transition memory and
  re-sampled for training, breaking the correlation between consecutive frames.
- **A target network** — a frozen copy of the network provides stable regression
  targets, synced every 1,000 training steps.

## The goal

The goal is not merely "a high score". The goal is to **understand what makes the agent
learn** — and to build a public, reproducible record of that understanding:

1. a working, inspectable implementation;
2. a series of numbered, documented experiments;
3. a growing knowledge base of observations and references;
4. a website that presents all of it honestly — including failures.

## The experiment

This project treats *itself* as an experiment. Changes are not silent edits: each
meaningful change — a new reward term, a different state variable, a sampling strategy —
is meant to become a numbered entry in `experiments/` with a hypothesis, a
configuration, results, and a conclusion, preserved at a Git tag.

```text
Train → observe → hypothesize → modify → train → evaluate → document → repeat
```

The agent is learning. The project is learning with it.
