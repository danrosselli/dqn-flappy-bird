---
layout: layouts/article.njk
title: Neural Network
description: The policy network — 8 → 64 → 64 → 2, 4,866 parameters, trained in the browser with TensorFlow.js.
section: how-it-works
sectionTitle: How It Works
collection: howItWorks
tags: how-it-works
order: 7
keywords: ["neural network", "deep q-network", "tensorflow.js model", "fully connected"]
scripts:
  - /assets/js/net.js
---

The Q-function is approximated by a fully-connected network defined in
`src/rl/dqn.js`. It runs on TensorFlow.js with the **WebGL backend**, so training
uses the GPU of whatever machine opens the page.

{% include "components/nn-canvas.njk" %}

## Architecture

<div class="table-wrap" markdown="0">
  <table>
    <thead>
      <tr><th scope="col">Layer</th><th scope="col">Units</th><th scope="col">Activation</th><th scope="col">Parameters</th></tr>
    </thead>
    <tbody>
      <tr><td>Input</td><td>{{ impl.network.input }}</td><td>—</td><td>—</td></tr>
      {% for layer in impl.network.layers %}
      <tr>
        <td>Hidden {{ loop.index }}</td>
        <td class="reward-value">{{ layer.units }}</td>
        <td>{{ layer.activation }}</td>
        <td class="reward-value">{{ layer.params | num }}</td>
      </tr>
      {% endfor %}
      <tr>
        <td>Output</td>
        <td class="reward-value">{{ impl.network.output.units }}</td>
        <td>{{ impl.network.output.activation }}</td>
        <td class="reward-value">{{ impl.network.output.params | num }}</td>
      </tr>
      <tr>
        <td><strong>Total</strong></td><td>—</td><td>—</td>
        <td class="reward-value"><strong>{{ impl.network.totalParams | num }}</strong></td>
      </tr>
    </tbody>
  </table>
</div>

**Input** — the 8-value [state vector]({{ '/how-it-works/state/' | url }}).
**Output** — two linear units: `Q(s, IDLE)` and `Q(s, FLAP)`. Linear activations are
required because Q-values are unbounded regression targets, not probabilities.

## Training configuration

- **Optimizer:** Adam, learning rate `0.001`
- **Loss:** mean squared error against the Bellman target
- **Batch:** 64 transitions, sampled uniformly from the legacy replay buffer
- **Schedule:** one update every 2 environment steps; a flag skips overlapping calls
  if the previous `fit` is still running

## Reading the Q-values

The two outputs are directly interpretable. When the bird approaches a pipe slightly
below the gap, a healthy network should price `FLAP` above `IDLE`; above the gap, the
reverse. The live demo HUD prints both values every frame — the difference between
them is, in a sense, the agent's confidence.
