---
layout: layouts/insight.njk
title: Target networks — stabilizing a moving target
description: Why the agent trains against a frozen copy of itself, and what the 1,000-step sync interval trades off.
section: insights
tags: insight
category: architecture
order: 4
updated: 2026-08-10
keyIdea: "Bootstrapping + function approximation + off-policy learning is the 'deadly triad'; the target network doesn't remove the danger, but it slows the feedback loop enough for training to converge in practice."
relatedExperiments: []
references:
  - authors: "V. Mnih, K. Kavukcuoglu, D. Silver, et al."
    title: "Human-level control through deep reinforcement learning"
    venue: "Nature, 518"
    year: 2015
    url: "https://doi.org/10.1038/nature14236"
    label: "doi:10.1038/nature14236"
---

## The moving-target problem

The Bellman target `r + γ·max Q(s′)` is computed *by the same network being trained*.
Every gradient step therefore changes both the predictions and the targets — the
regression chases its own tail. With linear function approximation this can still
converge; with a deep network it can spiral.

## The 2015 answer

Mnih et al.'s DQN introduced the now-standard fix: keep a second network, frozen, and
use it exclusively to compute targets. Every C steps, copy the online weights into it.
Between syncs, targets behave like a normal supervised-learning problem.

This project syncs every **1,000 training steps**. The trade-off is intuitive:

- **Sync too often** and you approximate the unstable no-target-network regime.
- **Sync too rarely** and targets lag behind what the online network has learned,
  slowing credit assignment for newly discovered good behavior.

## What would count as evidence

The right way to evaluate sync intervals here is not vibes: fix everything else,
vary `TARGET_UPDATE_FREQ`, and compare score-progression curves across matched episode
counts. The experiment infrastructure (`config.json` → `results.json` → Git tag)
exists precisely for this kind of single-variable question.
