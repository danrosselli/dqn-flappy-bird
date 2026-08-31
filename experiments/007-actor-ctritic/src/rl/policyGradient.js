/* ============================================================
 * ACTOR-CRITIC (Online TD) — FLAPPY BIRD
 * ------------------------------------------------------------
 * The Phaser game is the training environment itself.
 *
 * Two separate networks:
 *   - Actor:  outputs action probabilities (softmax)
 *   - Critic: estimates state value V(s) (linear output)
 *
 * Updates happen online — one gradient step per frame.
 * No trajectory buffer. Memory usage is O(1) regardless of
 * episode length, solving the memory issue of vanilla REINFORCE.
 *
 * Advantage: A = r + γ * V(s') - V(s)  (one-step TD error)
 * Actor loss:  -log π(a|s) * A
 * Critic loss: A²
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { PersistenceManager } from './persistenceManager.js';

export const ACTION_IDLE = 0;
export const ACTION_FLAP = 1;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];
export const STATE_SIZE = 8;
export const ACTION_SIZE = 2;

export const gamma = 0.99;
export const LEARNING_RATE = 0.001;
export const CRITIC_LEARNING_RATE = 0.002;

const POLICY_EPSILON = 1e-7;

export class ActorCriticAgent {
  constructor() {
    this.actor = this.createActor();
    this.critic = this.createCritic();
    this.persistence = new PersistenceManager();
    this.trainingInProgress = false;
    this.lastTrainingLoss = null;
    this.lastCriticLoss = null;

    // Previous step data for online update (replaces trajectory buffer)
    this.prevState = null;
    this.prevAction = null;
    this.prevValue = null;
    this.prevLogProb = null;
  }

  createActor() {
    const model = tf.sequential();

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      inputShape: [STATE_SIZE]
    }));

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    model.add(tf.layers.dense({
      units: ACTION_SIZE,
      activation: 'softmax'
    }));

    model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'categoricalCrossentropy'
    });

    return model;
  }

  createCritic() {
    const model = tf.sequential();

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      inputShape: [STATE_SIZE]
    }));

    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));

    model.add(tf.layers.dense({
      units: 1,
      activation: 'linear'
    }));

    model.compile({
      optimizer: tf.train.adam(CRITIC_LEARNING_RATE),
      loss: 'meanSquaredError'
    });

    return model;
  }

  resetEpisode() {
    this.prevState = null;
    this.prevAction = null;
    this.prevValue = null;
    this.prevLogProb = null;
  }

  /**
   * Samples an action from the actor and estimates V(s) from the critic.
   * Returns action, log probability, and value estimate.
   */
  chooseAction(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);

      // Actor forward pass
      const probabilities = this.actor.predict(stateTensor);
      const probs = probabilities.dataSync();

      // Sample action
      const random = Math.random();
      const action = random < probs[ACTION_FLAP]
        ? ACTION_FLAP
        : ACTION_IDLE;

      const logProb = Math.log(probs[action] + POLICY_EPSILON);

      // Critic forward pass
      const valueTensor = this.critic.predict(stateTensor);
      const value = valueTensor.dataSync()[0];

      return {
        action,
        idleProbability: probs[ACTION_IDLE],
        flapProbability: probs[ACTION_FLAP],
        logProb,
        value
      };
    });
  }

  /**
   * Online Actor-Critic update (one-step TD).
   *
   * Called every frame after the action is executed and the reward
   * is observed, along with the next state.
   *
   * 1. Compute advantage: A = r + γ * V(s') - V(s)
   * 2. Update critic:  minimize (target - V(s))²
   * 3. Update actor:   minimize -log π(a|s) * A
   */
  async update(prevState, prevAction, prevLogProb, prevValue, reward, nextState, done) {
    if (this.trainingInProgress) {
      return null;
    }

    this.trainingInProgress = true;

    try {
      const stateTensor = tf.tensor2d([prevState], [1, STATE_SIZE]);
      const nextStateTensor = tf.tensor2d([nextState], [1, STATE_SIZE]);

      // V(s') for advantage calculation
      const valueCurrent = this.critic.predict(nextStateTensor).dataSync()[0];
      const target = reward + (done ? 0 : gamma * valueCurrent);
      const advantage = target - prevValue;

      // --- Critic update: minimize (target - V(s))² ---
      // minimize() scopes gradients to only this model's variables.
      const criticLoss = this.critic.optimizer.minimize(() => {
        const v = this.critic.predict(stateTensor);
        return tf.losses.meanSquaredError(
          tf.tensor2d([[target]], [1, 1]), v
        ).mean();
      }, true);

      // --- Actor update: minimize -log π(a|s) * advantage ---
      const actorLoss = this.actor.optimizer.minimize(() => {
        const probs = this.actor.predict(stateTensor);
        const safeProbs = probs.add(POLICY_EPSILON);
        const logProbs = safeProbs.log();
        const actionMask = tf.oneHot(tf.tensor1d([prevAction], 'int32'), ACTION_SIZE);
        const selectedLogProb = logProbs.mul(actionMask).sum(1);
        return selectedLogProb.mul(-advantage).mean();
      }, true);

      this.lastTrainingLoss = actorLoss.dataSync()[0];
      this.lastCriticLoss = criticLoss.dataSync()[0];

      actorLoss.dispose();
      criticLoss.dispose();
      stateTensor.dispose();
      nextStateTensor.dispose();

      return this.lastTrainingLoss;
    } catch (error) {
      console.error('Actor-Critic update error:', error);
      return null;
    } finally {
      this.trainingInProgress = false;
    }
  }

  getPolicy(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const probabilities = this.actor.predict(stateTensor);
      const p = probabilities.dataSync();

      return {
        [ACTION_IDLE]: p[ACTION_IDLE],
        [ACTION_FLAP]: p[ACTION_FLAP]
      };
    });
  }

  getValue(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const v = this.critic.predict(stateTensor);
      return v.dataSync()[0];
    });
  }

  async saveBrain(generation, highScore = 0) {
    await this.persistence.saveActor(this.actor);
    await this.persistence.saveCritic(this.critic);
    await this.persistence.saveMetadata({
      algorithm: 'actor-critic',
      generation,
      highScore
    });
    console.log('Actor-Critic salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      const metadata = await this.persistence.loadMetadata();

      if (metadata && metadata.algorithm && metadata.algorithm !== 'actor-critic') {
        console.log('Memória antiga pertence a outro algoritmo. Iniciando Actor-Critic.');
        await this.persistence.clearAll();
        return { success: false, generation: 1 };
      }

      const actor = await this.persistence.loadActor();
      const critic = await this.persistence.loadCritic();

      if (actor && critic) {
        const actorInputSize = actor.inputs[0].shape[1];
        const actorOutputSize = actor.outputs[0].shape[1];
        const criticInputSize = critic.inputs[0].shape[1];
        const criticOutputSize = critic.outputs[0].shape[1];

        if (actorInputSize !== STATE_SIZE || actorOutputSize !== ACTION_SIZE ||
          criticInputSize !== STATE_SIZE || criticOutputSize !== 1) {
          console.log('Modelos salvos incompatíveis com Actor-Critic. Resetando.');
          await this.persistence.clearAll();
          return { success: false, generation: 1 };
        }

        this.actor = actor;
        this.actor.compile({
          optimizer: tf.train.adam(LEARNING_RATE),
          loss: 'categoricalCrossentropy'
        });

        this.critic = critic;
        this.critic.compile({
          optimizer: tf.train.adam(CRITIC_LEARNING_RATE),
          loss: 'meanSquaredError'
        });

        if (metadata) {
          const generation = metadata.generation ?? 1;
          const highScore = metadata.highScore ?? 0;
          console.log(
            'Actor-Critic carregado. Gen:',
            generation,
            'HighScore:',
            highScore
          );
          return { success: true, generation, highScore };
        }

        return { success: true, generation: 1, highScore: 0 };
      }
    } catch (error) {
      console.log('Nenhum estado de Actor-Critic salvo encontrado:', error.message);
    }

    return { success: false, generation: 1 };
  }
}

export async function resetBrain() {
  const pm = new PersistenceManager();
  await pm.clearAll();
  console.log('Memória do Actor-Critic resetada');
}