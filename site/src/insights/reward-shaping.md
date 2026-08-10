---
layout: layouts/insight.njk
title: Reward shaping — teaching by gradient, not verdict
description: Why this agent gets a continuous "how well am I aligned" signal instead of only pipe-or-death verdicts, and what that choice costs.
section: insights
tags: insight
category: reward
order: 1
updated: 2026-08-10
keyIdea: Dense, well-shaped rewards can turn an unlearnable sparse task into a tractable one — but every shaped term is a hypothesis about good behavior, and the agent will exploit whatever the hypothesis gets wrong.
relatedExperiments: []
references:
  - authors: "A. Y. Ng, D. Harada, S. Russell"
    title: "Policy invariance under reward transformations: Theory and application to reward shaping"
    venue: "Proceedings of the 16th International Conference on Machine Learning (ICML)"
    year: 1999
    url: "https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf"
    label: "people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf"
---

## The sparse-reward problem

The "natural" reward for Flappy Bird is sparse: +1 per pipe, death ends the episode.
For a tabula-rasa agent this is brutal — the probability of a random policy stumbling
into a pipe crossing is low, so almost every transition carries the same uninformative
survival signal. Learning signal density, not algorithm quality, becomes the
bottleneck.

## What this implementation does

The current reward function adds a **Gaussian proximity term**: reward decays smoothly
with the bird's distance from the gap center, shifted so that bad alignment is
actively negative. The agent gets graded *continuously*, every frame, on a quantity it
can directly control. Early training stops being a lottery and becomes a descent.

## The risks, honestly

- **Reward hacking.** Any shaped term can be gamed. A proximity reward could, in
  principle, teach the agent to hover near gap centers *without crossing pipes* if the
  crossing bonus isn't large enough to break the tie.
- **Conservatism.** A large death penalty makes "do nothing risky" locally optimal.
  The classic symptom: an agent that survives by camping low and refusing to commit
  to gaps. Watching for this failure mode is one reason the live HUD shows Q-values.
- **Non-invariance.** Ng, Harada & Russell proved that *potential-based* shaping
  preserves the optimal policy — and that arbitrary shaping (like the Gaussian used
  here) does not come with that guarantee. Shaping here is a deliberate, monitored
  trade: faster early learning in exchange for a subtly different objective.

## What to watch in experiments

If a future experiment compares shaped vs. sparse rewards, the comparison metrics
should include *episodes to first pipe* and *survival time distribution* — not just
best score — because shaping's effect shows up first in how quickly behavior becomes
non-random.
