---
layout: layouts/insight.njk
title: "The training flow — from game frame to weight update"
description: "The complete per-frame loop: observe, store, train, act — and the fine-grained mechanics of how a batch of 64 transitions becomes a model.fit call."
section: insights
tags: insight
category: training
order: 7
updated: 2026-08-27
keywords: ["training loop", "per-frame training", "batch training", "model fit", "DQN flow"]
keyIdea: "Every frame follows a strict sequence — observe, store, train, act — and the training step itself is a chain from random batch to Bellman target to single-output gradient update."
relatedExperiments: []
references:
  - authors: "V. Mnih, K. Kavukcuoglu, D. Silver, et al."
    title: "Human-level control through deep reinforcement learning"
    venue: "Nature, 518"
    year: 2015
    url: "https://doi.org/10.1038/nature14236"
    label: "doi:10.1038/nature14236"
---

## The game-side loop

The Phaser `update()` method (`src/game/scenes/Game.js`) runs every frame and
follows a strict five-step sequence:

```text
1. Observe currentState
2. If lastState/lastAction exist:
   - Compute reward from the previous transition
   - Store (lastState, lastAction, reward, currentState, done) in the buffer
   - Call agent.train()
3. Choose next action via epsilon-greedy
4. Execute the action (flap or idle)
5. Save currentState and action as lastState/lastAction
```

The transition storage looks like this in code:

```javascript
this.agent.replayBuffer.add(
  this.lastState,   // s
  this.lastAction,  // a
  reward,           // r
  currentState,     // s'
  this.gameOver     // done
);
await this.agent.train();
```

## The train() method: gatekeeping

Before any math happens, `train()` (`src/rl/dqn.js`) checks two conditions:

```text
1. Throttle:     stepCount % TRAIN_THROTTLE (2) === 0?
2. Minimum data:  replayBuffer.size() >= BATCH_SIZE (64)?
```

If either fails, the frame is skipped entirely. This prevents wasted computation
when the buffer is too small or when training every single frame would be
redundant.

## From buffer to batch

```javascript
const batch = this.replayBuffer.sampleRandomBasic(BATCH_SIZE);
```

Sixty-four transitions are sampled **uniformly at random** from the buffer. Each
transition contains:

```text
{ state, action, reward, nextState, done }
```

The batch is then split into parallel arrays — `states`, `actions`, `rewards`,
`nextStates`, `dones` — and converted to TensorFlow.js tensors.

## Computing the target

The Bellman target is built in four operations:

```text
1. nextQValues  = targetModel.predict(nextStateTensor)   // Q(s', ·) via frozen network
2. maxNextQ     = nextQValues.max(1)                      // max_a' Q(s', a')
3. notDone      = logicalNot(cast(doneTensor, 'bool'))    // mask for terminal states
4. targetQ      = reward + gamma * maxNextQ * notDone      // Bellman equation
```

The target network ensures these values are stable — they don't change while the
online network is being trained on this batch.

## The TD-error gate

Before committing to a `model.fit`, the code checks whether the batch is worth
training on:

```text
For each transition:
  currentQ = qValues[i][action]          // what the online network predicted
  target   = targetQ[i]                  // what Bellman says it should have been
  absError = |target - currentQ|
  normError = absError / (|target| + ε)

meanNormTDError = average(normErrors across batch)

if meanNormTDError >= TD_NORM_THRESHOLD (0.04):
    train the batch
else:
    skip — the network already predicts well enough
```

This is a batch-level heuristic: if the average normalized error is small, the
network already understands these transitions well enough, and training would add
noise rather than signal.

## The single-output update

The actual `model.fit` uses a clever trick. The network outputs two Q-values per
state: `[Q_IDLE, Q_FLAP]`. The target array is copied from the current predictions,
and **only the position of the taken action** is replaced with the Bellman target:

```text
Action taken = FLAP (index 1)

Before:  qValues = [3.2, 5.1]
Target:  target   = [3.2, 8.7]   // only index 1 changed

Loss:    MSE([3.2, 5.1], [3.2, 8.7]) → gradient pushes Q_FLAP toward 8.7
                                        Q_IDLE contributes ~0 loss
```

Over many batches, both outputs converge — but each gradient step only corrects
one of them.

## When the target network updates

Every **1,000 training steps**, the online network's weights are copied into the
target network:

```javascript
if (this.stepCount % TARGET_UPDATE_FREQ === 0) {
  this.targetModel.setWeights(this.model.getWeights());
}
```

Between syncs, the target network is frozen — a stable regression target for the
online network to chase.
