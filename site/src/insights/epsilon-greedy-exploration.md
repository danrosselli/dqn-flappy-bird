---
layout: layouts/insight.njk
title: Exploration is a design choice, not a constant
description: Reading the ε-greedy schedule of this project — and why the random branch flaps only 5% of the time.
section: insights
tags: insight
category: exploration
order: 2
updated: 2026-08-10
keywords: ["epsilon greedy", "exploration strategy", "bandit problem", "action selection"]
keyIdea: In environments where random actions are lethal, uniform ε-greedy exploration is silently domain-hostile; biasing the explore branch toward survivable actions injects domain knowledge without touching the learned policy.
relatedExperiments: []
references:
  - authors: "R. S. Sutton, A. G. Barto"
    title: "Reinforcement Learning: An Introduction (2nd ed.)"
    venue: "MIT Press"
    year: 2018
    url: "http://incompleteideas.net/book/the-book-2nd.html"
    label: "incompleteideas.net/book/the-book-2nd.html"
---

## The standard picture

ε-greedy is the simplest possible answer to exploration: with probability ε act
randomly, otherwise act greedily. Sutton & Barto cover it in a few pages precisely
because it is the baseline everything else is measured against. This project uses it
with a multiplicative decay: ε ← ε × 0.9995 per training step, from 0.9 toward 0.

## What the standard picture hides

Textbook ε-greedy assumes random actions are *harmless noise*. In Flappy Bird they are
not: uniform random flapping drives the bird into the ceiling almost immediately. An
explore step doesn't just add noise — it frequently ends the episode, which means:

- exploration transitions are dominated by early deaths;
- the memory fills with short, low-information trajectories;
- the agent rarely *randomly discovers* what crossing a pipe looks like.

## The biased-explore fix

This implementation's explore branch flaps with probability 0.05 and idles otherwise.
The result is a random walk that glides: survivable, pipe-reaching, signal-producing.
The learned policy is untouched — the bias lives only in the exploration sampler.

## Open questions for future experiments

- Should the flap bias decay with ε, or stay fixed?
- Does a crash-focused replay sampler reduce the need for biased exploration?
- Is ε = 0 the right floor, or should a small permanent exploration rate remain as
  insurance against policy stagnation as the world speeds up?
