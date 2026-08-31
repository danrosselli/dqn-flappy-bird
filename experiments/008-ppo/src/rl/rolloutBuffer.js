/* ============================================================
 * ROLLOUT BUFFER — PPO
 * ------------------------------------------------------------
 * Circular buffer that collects a fixed number of steps (128)
 * before triggering a PPO update. Stores transitions and
 * computes GAE-lambda advantages and discounted returns.
 *
 * Memory layout: contiguous typed arrays for cache efficiency.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';

export const STATE_SIZE = 8;

export class RolloutBuffer {
  /**
   * @param {number} capacity - Maximum number of steps to store
   */
  constructor(capacity = 128) {
    this.capacity = capacity;
    this.ptr = 0;

    this.states = new Float32Array(capacity * STATE_SIZE);
    this.actions = new Uint8Array(capacity);
    this.rewards = new Float32Array(capacity);
    this.values = new Float32Array(capacity);
    this.logProbs = new Float32Array(capacity);
    this.dones = new Uint8Array(capacity);
  }

  get size() {
    return this.ptr;
  }

  isReady() {
    return this.ptr >= this.capacity;
  }

  /**
   * Store one transition.
   * @param {number[]} state - 8-dim state vector
   * @param {number} action - 0 (IDLE) or 1 (FLAP)
   * @param {number} reward - Scalar reward
   * @param {number} value  - V(s) from critic
   * @param {number} logProb - log π(a|s) from actor
   * @param {boolean} done  - Whether this is a terminal step
   */
  add(state, action, reward, value, logProb, done) {
    if (this.ptr >= this.capacity) {
      throw new Error('RolloutBuffer overflow');
    }

    const offset = this.ptr * STATE_SIZE;
    for (let i = 0; i < STATE_SIZE; i++) {
      this.states[offset + i] = state[i];
    }

    this.actions[this.ptr] = action;
    this.rewards[this.ptr] = reward;
    this.values[this.ptr] = value;
    this.logProbs[this.ptr] = logProb;
    this.dones[this.ptr] = done ? 1 : 0;

    this.ptr++;
  }

  /**
   * Compute GAE-lambda advantages and discounted returns.
   * Must be called after the rollout is filled and before get().
   *
   * A_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...
   * where δ_t = r_t + γ V(s_{t+1}) - V(s_t)
   *
   * @param {number} lastValue - V(s_{T}) for bootstrap (0 if terminal)
   * @param {number} gamma - Discount factor
   * @param {number} lam - GAE lambda
   */
  computeAdvantages(lastValue, gamma = 0.99, lam = 0.95) {
    this.advantages = new Float32Array(this.capacity);
    this.returns = new Float32Array(this.capacity);

    let gae = 0;

    for (let t = this.ptr - 1; t >= 0; t--) {
      const nextValue = t === this.ptr - 1 ? lastValue : this.values[t + 1];
      const nextNonTerminal = this.dones[t] ? 0 : 1;

      const delta = this.rewards[t] + gamma * nextValue * nextNonTerminal - this.values[t];
      gae = delta + gamma * lam * nextNonTerminal * gae;

      this.advantages[t] = gae;
      this.returns[t] = gae + this.values[t];
    }
  }

  /**
   * Returns the collected data as tensors for PPO training.
   * Caller must dispose returned tensors.
   *
   * @returns {{ states, actions, advantages, returns, oldLogProbs }}
   */
  get() {
    if (this.ptr === 0) {
      throw new Error('RolloutBuffer is empty');
    }

    const n = this.ptr;

    const states = tf.tensor2d(this.states.subarray(0, n * STATE_SIZE), [n, STATE_SIZE]);
    const actions = tf.tensor1d(Array.from(this.actions.subarray(0, n)), 'int32');
    const advantages = tf.tensor1d(this.advantages.subarray(0, n));
    const returns = tf.tensor1d(this.returns.subarray(0, n));
    const oldLogProbs = tf.tensor1d(this.logProbs.subarray(0, n));

    return { states, actions, advantages, returns, oldLogProbs };
  }

  /**
   * Get batch size (number of stored steps).
   */
  get size() {
    return this.ptr;
  }

  /**
   * Clear the buffer for the next rollout.
   */
  clear() {
    this.ptr = 0;
    this.advantages = null;
    this.returns = null;
  }
}
