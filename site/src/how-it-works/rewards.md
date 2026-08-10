---
layout: layouts/article.njk
title: Rewards
description: The reward function — survival, Gaussian gap proximity, velocity discipline, pipe bonuses, and the death penalty.
section: how-it-works
sectionTitle: How It Works
collection: howItWorks
tags: how-it-works
order: 5
---

The reward function is the only supervision the agent ever receives. Every behavior
the bird exhibits — cautious, aggressive, rhythmic — is an emergent consequence of
these numbers.

{% include "components/reward-table.njk" %}

## The shaping term, precisely

The dominant per-frame signal is a **Gaussian proximity reward** centered on the gap:

```text
normalizedDy  = |bird.y − gapCenter.y| / (screenHeight / 2)
proximity     = exp(−normalizedDy² / (2σ²)) − 0.35      with σ = 0.5
proximity     = max(proximity, −1.0)
```

Three design choices matter:

- **It is smooth.** Reward decreases continuously with distance from the gap center,
  giving the agent gradient-like guidance instead of sparse pipe-or-death feedback.
- **It crosses zero.** The `−0.35` shift means being *badly* aligned is actively
  punished, not merely unrewarded. Hovering far from the gap costs reward every frame.
- **It is clamped.** The penalty bottoms out at `−1.0`, so a single bad frame can
  never dominate an episode's return.

## The terminal transition

Death is special. The stored transition pairs the final state and action with
`reward = −20` and a **zeroed next state** marked `done`. Because the bootstrap term
`max Q(s′)` is masked out for terminal transitions, the target for the last action is
exactly `−20` — an unambiguous "never do this here again".

## What this function encourages

Read as a whole, the function says: *stay alive, stay near the gap center, don't
oscillate, and cash in by crossing pipes.* The survival term keeps episodes long; the
shaping term makes them informative; the bonus keeps score-chasing worthwhile; the
death penalty makes caution rational.
