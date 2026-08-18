---
layout: layouts/article.njk
title: Training Process
description: The per-frame training loop — observe, act, store, sample, update — exactly as it runs inside the Phaser scene.
section: training
sectionTitle: Training
collection: training
tags: training
order: 1
keywords: ["training loop", "per-frame training", "batch sampling", "dqn training"]
---

There is no separate training script. Training happens **inside the game loop**, one
slice per rendered frame, interleaved with acting. This is the exact sequence that
`Game.update()` executes, at up to 60 iterations per second:

{% include "components/training-loop.njk" %}

## Throughput discipline

Training on a GPU from a per-frame game loop needs throttling, or the frame rate
collapses under its own weight. Two mechanisms keep the loop responsive:

- **Train throttle** — an optimization step runs only every **2** environment steps.
- **Re-entrancy guard** — if the previous `model.fit` is still running, the next call
  is skipped instead of queued. Training never blocks the game; the game never waits
  for training.

## Where the time goes

Each training step samples 64 transitions, builds tensors, computes Bellman targets
with the target network, and runs one Adam update — all inside `tf.tidy` scopes so
intermediate tensors are disposed. Memory leaks in this loop would slowly strangle
the tab, so tensor lifecycle is managed as carefully as the learning itself.

## Episodes are generations

One episode = one life. Death persists the brain and increments the **generation**
counter, so "Gen" on the HUD is literally "how many times the agent has died and
restarted". Early training burns through generations in seconds; a competent agent
stretches each one for minutes.
