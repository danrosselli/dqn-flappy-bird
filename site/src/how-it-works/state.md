---
layout: layouts/article.njk
title: State
description: The 8 normalized values the agent receives every frame — pipe distances, gap geometry, velocity, and world speed.
section: how-it-works
sectionTitle: How It Works
collection: howItWorks
tags: how-it-works
order: 3
---

The agent sees the world as a vector of **8 continuous, normalized values**, rebuilt
every frame. There is no discretization — the raw geometry of the game goes straight
into the network.

{% include "components/state-vector.njk" %}

## Variable by variable

{% for item in impl.state %}
### `{{ item.key }}` — {{ item.label }}

{{ item.description }}

<div class="chip-row" markdown="0">
  <span class="chip">formula · {{ item.formula }}</span>
  <span class="chip">range · {{ item.range }}</span>
</div>
{% endfor %}

## Design notes

**Two pipes, not one.** The state describes the current pipe pair *and* the next one
(`dxNext`, `dyNext`, `gapNext`). A single-pipe state forces the agent to react to each
obstacle in isolation; the lookahead lets it position for the gap after this one.

**Speed is observable.** Because scroll speed grows with the score, the same geometry
can demand different timing at different speeds. Including `speed` makes the state
(approximately) Markovian again.

**Normalization is approximate, on purpose.** Dividers are physical scales of the world
— screen width, gravity, the gap range — not per-episode statistics. Most values land in
roughly `0 … 1`; excursions outside are survivable because the network was never
promised a strict range.

**Empty frames are discarded.** When no pipe is on screen, `dx` normalizes to `1.0`.
The replay buffer refuses to store those frames — they contain no obstacle information
and would dilute the memory with noise.
