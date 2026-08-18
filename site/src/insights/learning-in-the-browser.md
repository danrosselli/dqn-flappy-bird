---
layout: layouts/insight.njk
title: The browser as a training environment
description: What it means to train a DQN inside a game tab — WebGL compute, IndexedDB persistence, throttling, and the constraints this imposes on rigor.
section: insights
tags: insight
category: training
order: 5
updated: 2026-08-10
keywords: ["browser ai", "tensorflow.js", "client-side ml", "webgl compute"]
keyIdea: "Running RL in the browser trades reproducibility for accessibility — anyone's GPU becomes a training node, but wall-clock pacing, tab lifecycle, and unseeded randomness make strict experiment protocol a discipline, not a default."
relatedExperiments: []
references:
  - authors: "D. Smilkov, N. Thorat, Y. Assogba, et al."
    title: "TensorFlow.js: Machine Learning for the Web and Beyond"
    venue: "Proceedings of Machine Learning and Systems (MLSys)"
    year: 2019
    url: "https://arxiv.org/abs/1901.05350"
    label: "arxiv.org/abs/1901.05350"
---

## What the browser buys

Zero install, GPU access through WebGL, and a game engine already living there.
The entire pipeline — rendering, physics, inference, gradient updates, persistence —
fits in one tab, and TensorFlow.js makes `model.fit` on the GPU feel unremarkable.

## What it costs

- **Wall-clock coupling.** Training progresses at frame rate, not at compute speed.
  "Episodes per hour" depends on the tab being visible and the machine being awake.
- **No seeds, no clones.** `Math.random()` drives exploration and pipe spawns; runs
  are not bitwise reproducible. Comparisons between experiments need aggregated
  metrics, not single-run score screenshots.
- **Tab lifecycle is part of the protocol.** Background tabs get throttled by the
  browser; IndexedDB writes are async and can race a closing tab. The save-on-death
  design is a deliberate answer: persist at the moment that matters most.

## Discipline that compensates

The project compensates with structure rather than runtime guarantees: numbered
experiments, machine-readable configs, recorded hyperparameters, and Git tags. The
browser makes training casual; the experiment log makes it count.
