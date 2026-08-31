/* ============================================================
 * POLICY GRADIENT (REINFORCE) — FLAPPY BIRD
 * ------------------------------------------------------------
 * The Phaser game is the training environment itself.
 *
 * There is no replay buffer and no target network.
 * A complete episode is collected as a trajectory and, when the
 * bird dies, one policy-gradient update is performed with all
 * state/action/reward steps from that episode.
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

const POLICY_EPSILON = 1e-7;

export class PolicyGradientAgent {
  constructor() {
    this.model = this.createModel();
    this.persistence = new PersistenceManager();
    this.trajectory = [];
    this.trainingInProgress = false;
    this.lastTrainingLoss = null;
    this.lastEpisodeSteps = 0;
  }

  createModel() {
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

    // Policy: each output is the probability of an action.
    model.add(tf.layers.dense({
      units: ACTION_SIZE,
      activation: 'softmax'
    }));

    // compile is kept so TensorFlow.js treats the model as trainable
    // and so the optimizer is persisted/recreated consistently.
    model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'categoricalCrossentropy'
    });

    return model;
  }

  resetEpisode() {
    this.trajectory = [];
    this.lastEpisodeSteps = 0;
  }

  recordStep(state, action, reward) {
    this.trajectory.push({
      state: [...state],
      action,
      reward
    });
  }

  /**
   * Samples an action from the policy distribution.
   *
   * Example:
   *   [0.30, 0.70] -> FLAP with 70% probability.
   */
  chooseAction(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const probabilities = this.model.predict(stateTensor);
      const probs = probabilities.dataSync();

      const random = Math.random();
      const action = random < probs[ACTION_FLAP]
        ? ACTION_FLAP
        : ACTION_IDLE;

      return {
        action,
        idleProbability: probs[ACTION_IDLE],
        flapProbability: probs[ACTION_FLAP]
      };
    });
  }

  /**
   * REINFORCE:
   *   1. Calculate discounted returns G_t for the complete trajectory.
   *   2. Normalize returns for a more stable update.
   *   3. Minimize -G_t * log(pi(a_t | s_t)).
   *   4. One backpropagation/update is performed for the episode.
   */
  async train() {
    if (this.trainingInProgress || this.trajectory.length === 0) {
      return null;
    }

    this.trainingInProgress = true;

    const trajectory = this.trajectory;
    this.lastEpisodeSteps = trajectory.length;

    const states = trajectory.map(step => step.state);
    const actions = trajectory.map(step => step.action);
    const rewards = trajectory.map(step => step.reward);

    // Once train starts, the old trajectory is no longer needed.
    // Keep local references for this update and clear the agent's buffer.
    this.trajectory = [];

    try {
      const returns = new Array(rewards.length);
      let runningReturn = 0;

      for (let i = rewards.length - 1; i >= 0; i--) {
        runningReturn = rewards[i] + gamma * runningReturn;
        returns[i] = runningReturn;
      }

      const stateTensor = tf.tensor2d(states, [states.length, STATE_SIZE]);
      const actionTensor = tf.tensor1d(actions, 'int32');
      const returnTensor = tf.tensor1d(returns);

      const advantages = tf.tidy(() => {
        const mean = returnTensor.mean();
        const std = returnTensor.sub(mean).square().mean().sqrt().add(1e-8);
        return returnTensor.sub(mean).div(std);
      });

      const { value, grads } = tf.variableGrads(() => {
        const probabilities = this.model.apply(stateTensor);
        const safeProbabilities = probabilities.add(POLICY_EPSILON);
        const logProbabilities = safeProbabilities.log();
        const actionMask = tf.oneHot(actionTensor, ACTION_SIZE);
        const selectedLogProbabilities = logProbabilities.mul(actionMask).sum(1);

        // REINFORCE loss: - G_t * log(pi(a_t | s_t))
        return selectedLogProbabilities
          .mul(advantages)
          .neg()
          .mean();
      });

      this.model.optimizer.applyGradients(grads);

      this.lastTrainingLoss = value.dataSync()[0];

      value.dispose();
      Object.values(grads).forEach(gradient => gradient.dispose());
      advantages.dispose();
      returnTensor.dispose();
      actionTensor.dispose();
      stateTensor.dispose();

      return this.lastTrainingLoss;
    } catch (error) {
      console.error('Policy Gradient training error:', error);
      // Do not silently retain a trajectory that has already been consumed.
      return null;
    } finally {
      this.trainingInProgress = false;
    }
  }

  getPolicy(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const probabilities = this.model.predict(stateTensor);
      const p = probabilities.dataSync();

      return {
        [ACTION_IDLE]: p[ACTION_IDLE],
        [ACTION_FLAP]: p[ACTION_FLAP]
      };
    });
  }

  async saveBrain(generation, highScore = 0) {
    await this.persistence.saveModel(this.model);
    await this.persistence.saveMetadata({
      algorithm: 'reinforce',
      generation,
      highScore
    });
    console.log('Policy Gradient salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      const metadata = await this.persistence.loadMetadata();

      // Never load the previous DQN into this policy-gradient agent.
      if (metadata && metadata.algorithm && metadata.algorithm !== 'reinforce') {
        console.log('Memória antiga pertence a outro algoritmo. Iniciando Policy Gradient.');
        await this.persistence.clearAll();
        return { success: false, generation: 1 };
      }

      const model = await this.persistence.loadModel();

      if (model) {
        const modelInputSize = model.inputs[0].shape[1];
        const outputSize = model.outputs[0].shape[1];

        if (modelInputSize !== STATE_SIZE || outputSize !== ACTION_SIZE) {
          console.log('Modelo salvo incompatível com Policy Gradient. Resetando.');
          await this.persistence.clearAll();
          return { success: false, generation: 1 };
        }

        this.model = model;
        this.model.compile({
          optimizer: tf.train.adam(LEARNING_RATE),
          loss: 'categoricalCrossentropy'
        });

        if (metadata) {
          const generation = metadata.generation ?? 1;
          const highScore = metadata.highScore ?? 0;
          console.log(
            'Policy Gradient carregado. Gen:',
            generation,
            'HighScore:',
            highScore
          );
          return { success: true, generation, highScore };
        }

        return { success: true, generation: 1, highScore: 0 };
      }
    } catch (error) {
      console.log('Nenhum estado de Policy Gradient salvo encontrado:', error.message);
    }

    return { success: false, generation: 1 };
  }
}

export async function resetBrain() {
  const pm = new PersistenceManager();
  await pm.clearAll();
  console.log('Memória do Policy Gradient resetada');
}
