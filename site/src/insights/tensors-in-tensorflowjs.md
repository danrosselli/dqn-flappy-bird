---
layout: layouts/insight.njk
title: "Tensors in TensorFlow.js — the currency of neural networks"
description: Why every input and output must be a tensor, how arrays are converted, and what WebGL GPU acceleration buys in a browser-based DQN.
section: insights
tags: insight
category: performance
order: 9
updated: 2026-08-27
keywords: ["tensor", "tensorflow js", "webgl", "GPU", "browser ML", "data conversion"]
keyIdea: "TensorFlow.js speaks tensors — all data must cross the Array↔Tensor boundary at input and output, and the conversion is what unlocks WebGL GPU acceleration."
relatedExperiments: []
references:
  - authors: "TensorFlow.js Team"
    title: "TensorFlow.js — Machine Learning for JavaScript"
    venue: "tensorflow.org"
    year: 2024
    url: "https://www.tensorflow.org/js"
    label: "tensorflow.org/js"
---

## Why tensors, not arrays

TensorFlow.js cannot operate on plain JavaScript arrays. Every operation inside the
neural network — matrix multiplication, activation functions, backpropagation —
requires **tensors**: typed, shaped, GPU-resident data containers.

The conversion boundary is simple:

```text
JavaScript array  →  tf.tensor2d(...)  →  tensor (enters the network)
Tensor            →  .dataSync()       →  TypedArray (exits the network)
```

## The conversion points in this project

**Choosing an action** — a single state enters the network:

```javascript
const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);  // [1, 8]
const qValues = this.model.predict(stateTensor);
```

**Training a batch** — 64 states, next-states, rewards, and dones:

```javascript
const stateTensor     = tf.tensor2d(states, [BATCH_SIZE, STATE_SIZE]);     // [64, 8]
const nextStateTensor = tf.tensor2d(nextStates, [BATCH_SIZE, STATE_SIZE]); // [64, 8]
const rewardTensor    = tf.tensor1d(rewards);                               // [64]
const doneTensor      = tf.tensor1d(dones);                                 // [64]
```

**Reading results back** — extracting scalars or arrays:

```javascript
.dataSync()    // synchronous, returns TypedArray
.arraySync()   // returns plain JavaScript array
```

## Array vs tensor at a glance

| Aspect | JavaScript Array | Tensor |
|---|---|---|
| Memory location | CPU | CPU or GPU (WebGL) |
| Speed for bulk math | Slow | Much faster with GPU |
| Operations | Manual loops | Optimized (`matMul`, `add`, `relu`…) |
| Auto-differentiation | No | Yes (autograd) |
| Shape | Flexible, untyped | Fixed, typed |
| Memory management | Garbage collected | Manual (`.dispose()`) |

## What the GPU actually buys

The backend is **WebGL** when available. This means:

- Batch matrix multiplications run on the GPU's parallel compute units
- A batch of 64 states is processed in one forward pass, not 64 sequential loops
- Backpropagation leverages the same parallel hardware

In practice, the per-frame overhead of tensor conversion is negligible compared to
the speedup from GPU-accelerated matrix operations during training.

## The tf.tidy pattern

Throughout the code, inference is wrapped in `tf.tidy()`:

```javascript
return tf.tidy(() => {
  const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
  const qValues = this.model.predict(stateTensor);
  return qValues.argMax(1).dataSync()[0];
});
```

`tf.tidy` automatically disposes all intermediate tensors created inside the
callback, preventing memory leaks. Without it, every `predict()` call would leak
tensor memory until the browser tab crashed.
