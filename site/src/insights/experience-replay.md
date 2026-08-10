---
layout: layouts/insight.njk
title: Experience replay — learning from the past without drowning in it
description: Why replay memory exists, what this project's three-buffer structure buys, and where prioritization might matter.
section: insights
tags: insight
category: replay
order: 3
updated: 2026-08-10
keyIdea: "Replay decorrelates training data and multiplies the value of rare events — but 'which past deserves to be re-sampled' is its own research question, which is why this codebase keeps six samplers alive."
relatedExperiments: []
references:
  - authors: "L.-J. Lin"
    title: "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching"
    venue: "Machine Learning, 8"
    year: 1992
    url: "https://link.springer.com/article/10.1007/BF00992699"
    label: "doi:10.1007/BF00992699"
  - authors: "T. Schaul, J. Quan, I. Antonoglou, D. Silver"
    title: "Prioritized Experience Replay"
    venue: "International Conference on Learning Representations (ICLR)"
    year: 2016
    url: "https://arxiv.org/abs/1511.05952"
    label: "arxiv.org/abs/1511.05952"
---

## Why replay at all

Lin introduced experience replay in 1992 for reasons that are still the reasons:
consecutive frames are nearly identical, so training on them in order violates the
i.i.d. assumptions that make stochastic gradient descent behave, and each experience
is used once and discarded — a terrible deal when the interesting ones are rare.

In this project the interesting events really are rare: +10 pipe crossings and −20
deaths are drops in a sea of +0.1 survivals. Replay gives those drops hundreds of
chances to matter.

## The three-buffer structure

The `ReplayBuffer` keeps a 50k circular buffer (the training source), a 10k recent
buffer (what just happened), and a 40k reservoir (a uniform long-term memory with a
guaranteed minimum replacement chance). The design tension being managed:

- **Recency** matters because the policy is changing — old transitions were produced
  by a worse agent.
- **Long memory** matters because competence at speed 200 shouldn't be forgotten by
  the time the world reaches 400.

## Where prioritization could enter

Schaul et al.'s prioritized replay samples transitions by TD-error magnitude — learn
from what surprises you. The codebase already contains a reward-magnitude variant and
a crash-focus variant as study artifacts. Whether either beats uniform sampling here
is an empirical question, and a clean candidate for an early numbered experiment:
identical everything, different sampler, comparable scores.
