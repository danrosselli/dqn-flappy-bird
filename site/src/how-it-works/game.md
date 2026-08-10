---
layout: layouts/article.njk
title: The Game
description: The Flappy Bird environment the agent plays — physics, pipes, scoring, and death, exactly as implemented with Phaser 3.
section: how-it-works
sectionTitle: How It Works
collection: howItWorks
tags: how-it-works
order: 1
---

The environment is a Flappy Bird game built with **Phaser 3.90** and its Arcade physics
engine, running on a 1024 × 768 canvas. It plays itself: there is no human input.
Every frame is one step of the learning loop.

{% set caption = "The environment, annotated with the quantities the agent observes." %}
{% include "components/game-figure.njk" %}

## The physics

| Constant | Value | Effect |
| --- | --- | --- |
| Gravity | `1000 px/s²` | Constant downward acceleration on the bird |
| Flap impulse | `−350 px/s` | Instantly sets vertical velocity upward |
| Flap gate | `velY > −180` | Flapping while already rising fast is ignored — no climb-spam |
| Bird position | `x = 120` | The bird never moves horizontally; the world scrolls |
| Tilt | `−20° … +20°` | Visual only: nose up on flap, dives as it falls |

## The pipes

A new pipe pair spawns every **1.9 seconds** just past the right edge of the screen.
Each pair has a random gap:

- **Gap height:** uniform between 200 and 410 px
- **Gap center:** uniform between 150 px and `height − 150`

The scroll speed is **not constant**:

```text
speed = min(200 + 0.4 × score, 400)   px/s
```

Every point earned makes the world slightly faster, up to a 400 px/s cap. High scores
are therefore genuinely harder than low scores — the policy that works at speed 200
is not the policy that works at speed 400, which is why speed is part of the
[state vector]({{ '/how-it-works/state/' | url }}).

## Scoring

Each pipe pair carries an invisible sensor zone at its **trailing edge**. When the bird
overlaps it, the score increments and a **+10** reward bonus is queued for the agent.
The sensor is consumed on contact, so a pipe can only be scored once.

## Death

An episode ends when the bird:

- collides with a pipe, or
- leaves the vertical bounds of the screen (`y < −50` or `y > height + 50`).

Death stores a terminal transition with reward **−20**, persists the agent's brain to
IndexedDB, increments the generation counter, and restarts the scene after 500 ms.
A new episode — a new *generation* — begins immediately, with the agent slightly
less ignorant than before.
