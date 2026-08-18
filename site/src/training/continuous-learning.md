---
layout: layouts/article.njk
title: Continuous Learning
description: The agent is never "done" — and neither is the project. The experiment lifecycle that turns observations into new training runs.
section: training
sectionTitle: Training
collection: training
tags: training
order: 4
keywords: ["continuous learning", "indexeddb", "persistent training", "browser training"]
---

Most game-AI demos train once, record a video, and stop. This project is built around
the opposite premise: **training is the steady state, not a phase.** That premise
operates at two levels.

## Level 1 — the agent never stops learning

Persistence turns a browser tab into a long-running training environment:

- every death saves the full state — weights, replay memory, ε, generation — to
  IndexedDB;
- every page load restores it;
- ε anneals toward zero, but the replay buffer keeps accepting experience and the
  network keeps updating.

An agent that played yesterday resumes today with yesterday's knowledge. "The current
brain" is a living artifact, not a checkpoint that was frozen for a demo.

## Level 2 — the project never stops experimenting

The outer loop belongs to the human. Watching the agent suggests a hypothesis; the
hypothesis becomes a numbered experiment; the experiment's result seeds the next one.

<ol class="steps" aria-label="The experiment lifecycle">
  <li class="step"><span class="step-num" aria-hidden="true">01</span><span class="step-body"><span class="step-name">Train</span><span class="step-detail">Run the current configuration until behavior stabilizes.</span></span></li>
  <li class="step"><span class="step-num" aria-hidden="true">02</span><span class="step-body"><span class="step-name">Observe</span><span class="step-detail">Watch the agent. Where does it die? What does it over-value?</span></span></li>
  <li class="step"><span class="step-num" aria-hidden="true">03</span><span class="step-body"><span class="step-name">Hypothesize</span><span class="step-detail">Form one testable claim — one variable, one expectation.</span></span></li>
  <li class="step"><span class="step-num" aria-hidden="true">04</span><span class="step-body"><span class="step-name">Modify</span><span class="step-detail">Change the code on a branch; keep everything else constant.</span></span></li>
  <li class="step"><span class="step-num" aria-hidden="true">05</span><span class="step-body"><span class="step-name">Train again</span><span class="step-detail">Same protocol, fresh brain, comparable conditions.</span></span></li>
  <li class="step"><span class="step-num" aria-hidden="true">06</span><span class="step-body"><span class="step-name">Evaluate & document</span><span class="step-detail">Results and conclusion into experiments/NNN/, tagged experiment-NNN.</span></span></li>
  <li class="step step--loop"><span class="step-num" aria-hidden="true">↺</span><span class="step-body"><span class="step-name">Repeat</span><span class="step-detail">Merge if it worked, keep the tag either way. Failures stay on the record.</span></span></li>
</ol>

## Why document failures?

An experiment that made the agent worse is not wasted work — it is a measurement of
the idea's effect, preserved so it is never tried twice by accident. The
[Experiments]({{ '/experiments/' | url }}) log is designed to hold both outcomes with
equal dignity.

The agent is learning. The project is learning with it.
