---
layout: layouts/article.njk
title: Actions
description: The two actions available to the agent — IDLE and FLAP — and the physics gate that constrains them.
section: how-it-works
sectionTitle: How It Works
collection: howItWorks
tags: how-it-works
order: 4
keywords: ["action space", "flap idle", "dqn actions", "binary actions"]
---

The action space is as small as reinforcement learning allows: **two discrete actions**.
Every frame, the agent picks exactly one.

<div class="table-wrap" markdown="0">
  <table>
    <thead>
      <tr><th scope="col">ID</th><th scope="col">Action</th><th scope="col">Effect</th></tr>
    </thead>
    <tbody>
      {% for action in impl.actions %}
      <tr>
        <td class="reward-value">{{ action.id }}</td>
        <td><strong>{{ action.name }}</strong></td>
        <td>{{ action.description }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

## Why a gate on flapping?

A flap is not "go up a bit" — it is a hard assignment of velocity to `−350 px/s`.
Without a constraint, an agent that learns "flapping is often good" can pin the bird
to the ceiling by flapping every frame. The implementation therefore ignores flap
requests while the bird is already rising faster than `−180 px/s`.

The gate quietly shapes the problem:

- the *effective* action near the ceiling is always IDLE;
- flapping too early wastes the impulse — the next request is gated until the bird
  slows down;
- the policy must learn **rhythm**, not just direction.

## Small action space, real consequences

Two actions do not make the problem trivial. The optimal action depends on the full
state: distance to the pipe, vertical offset, current velocity, gap size, *and* world
speed. A flap that is correct at 200 px/s can be fatal at 400 px/s.

The network outputs one Q-value per action, so deciding is an `argmax` over two
numbers — but learning what those numbers should be is the entire project.
