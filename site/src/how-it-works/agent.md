---
layout: layouts/article.njk
title: The Agent
description: The DQN agent — how it perceives the game, decides between IDLE and FLAP, and learns from every frame.
section: how-it-works
sectionTitle: How It Works
collection: howItWorks
tags: how-it-works
order: 2
---

The agent is a **Deep Q-Network** implemented with TensorFlow.js (WebGL backend).
It is not a scripted player and it receives no privileged information — only the same
kind of numbers a player could see: where the pipes are, how fast everything moves,
and how the bird is flying.

## The contract with the environment

Once per animation frame, four things happen in order:

1. **Observe** — the game builds an 8-value state vector from the world.
2. **Learn** — the previous transition `(s, a, r, s′, done)` is stored, and a training
   step may run against a sampled batch.
3. **Decide** — the agent picks `IDLE` or `FLAP`, ε-greedily.
4. **Act** — the action is applied to the bird and physics advances.

{% include "components/loop-diagram.njk" %}

## What the agent actually computes

Given a state, the network outputs **two numbers**: the estimated value of doing
nothing, `Q(s, IDLE)`, and the estimated value of flapping, `Q(s, FLAP)`. Acting is
just `argmax`. The in-game HUD shows both Q-values live, next to the chosen action —
you can watch the agent's preferences flip as pipes approach.

## Exploration is not uniform randomness

With probability ε the agent explores instead of exploiting. But exploration here is
deliberately biased: a "random" action flaps only **5%** of the time. Uniformly random
flapping kills the bird almost instantly — biased exploration produces random
trajectories that live long enough to contain useful signal.

## Learning continues across sessions

On every death the agent persists its complete state to the browser's IndexedDB:
network weights, replay buffers, current ε, and the generation counter. Reload the
page and training resumes — the experiment survives the tab that hosts it.
