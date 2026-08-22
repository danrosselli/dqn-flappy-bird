/* ============================================================
 * DEEP Q-NETWORK — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Uses TensorFlow.js to implement DQN with a neural network.
 * States are continuous [dx, dy, velY], no discretization.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { ReplayBuffer } from './replayBuffer.js';
import { PersistenceManager } from './persistenceManager.js';

/* ------------------------------------------------------------
 * ACTIONS
 * ------------------------------------------------------------ */
export const ACTION_FLAP = 1;
export const ACTION_IDLE = 0;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];
export const STATE_SIZE = 8;

/* ------------------------------------------------------------
 * HYPERPARAMETERS
 * ------------------------------------------------------------ */
export const gamma = 0.99;
export let epsilon = 0.9;
export const EPSILON_MIN = 0.000;
export const EPSILON_DECAY = 0.9995;
export const BATCH_SIZE = 64;
export const TARGET_UPDATE_FREQ = 1000;
export const LEARNING_RATE = 0.001;
export const TRAIN_THROTTLE = 2;

// --- Prioritized Training / TD-Error Skip (versão simples) ---
export const TD_ERROR_THRESHOLD = 0.20;   // ajuste este valor

/* ------------------------------------------------------------
 * DQN AGENT
 * ------------------------------------------------------------ */
export class DQNAgent {
  constructor() {
    // Force WebGL backend for better performance and stability
    tf.setBackend('webgl')
      .then(() => console.log('TensorFlow.js backend: WebGL'))
      .catch(err => console.warn('Failed to set WebGL:', err));

    this.model = this.createModel();
    this.targetModel = this.createModel();
    this.targetModel.setWeights(this.model.getWeights());

    this.replayBuffer = new ReplayBuffer();
    this.stepCount = 0;
    this.trainingInProgress = false;

    // Estado do Prioritized Training
    this.batchesSinceLastTrain = 0;
    this.lastTDError = 0;

    // Contadores para o HUD
    this.totalTrainAttempts = 0;
    this.totalTrained = 0;

    this.persistence = new PersistenceManager();
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

    model.add(tf.layers.dense({
      units: 2,
      activation: 'linear'
    }));

    model.compile({
      optimizer: tf.train.adam(LEARNING_RATE),
      loss: 'meanSquaredError'
    });

    return model;
  }

  async chooseAction(state) {
    if (Math.random() < epsilon) {
      return Math.random() < 0.05 ? ACTION_FLAP : ACTION_IDLE;
    }

    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const qValues = this.model.predict(stateTensor);
      const action = qValues.argMax(1).dataSync()[0];
      return action;
    });
  }

  async train() {
    this.stepCount++;

    // Throttle
    if (this.stepCount % TRAIN_THROTTLE !== 0) {
      return;
    }

    if (this.replayBuffer.size() < BATCH_SIZE) return;

    if (this.trainingInProgress) {
      console.log('Training skipped: Already in progress');
      return;
    }

    this.trainingInProgress = true;

    // Amostra o batch
    const batch = this.replayBuffer.sampleRandomBasic(BATCH_SIZE);

    const states = [];
    const actions = [];
    const rewards = [];
    const nextStates = [];
    const dones = [];

    batch.forEach(transition => {
      states.push(transition.state);
      actions.push(transition.action);
      rewards.push(transition.reward);
      nextStates.push(transition.nextState);
      dones.push(transition.done ? 1 : 0);
    });

    const stateTensor = tf.tensor2d(states, [BATCH_SIZE, STATE_SIZE]);
    const nextStateTensor = tf.tensor2d(nextStates, [BATCH_SIZE, STATE_SIZE]);
    const rewardTensor = tf.tensor1d(rewards);
    const doneTensor = tf.tensor1d(dones);

    try {
      // 1. Calcula targets + TD-error médio
      const { targets, meanTDError } = tf.tidy(() => {
        const nextQValues = this.targetModel.predict(nextStateTensor);
        const maxNextQ = nextQValues.max(1);
        const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
        const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));

        const qValues = this.model.predict(stateTensor);
        const qValuesArray = qValues.arraySync();
        const targetQArray = targetQ.arraySync();

        let totalAbsError = 0;
        for (let i = 0; i < BATCH_SIZE; i++) {
          const currentQ = qValuesArray[i][actions[i]];
          const target = targetQArray[i];
          totalAbsError += Math.abs(target - currentQ);
          qValuesArray[i][actions[i]] = target;
        }

        const meanTDError = totalAbsError / BATCH_SIZE;
        const targetsTensor = tf.tensor2d(qValuesArray, [BATCH_SIZE, 2]);

        nextQValues.dispose();
        maxNextQ.dispose();
        targetQ.dispose();
        qValues.dispose();

        return { targets: targetsTensor, meanTDError };
      });

      // 2. Decide só com base no TD-error
      this.totalTrainAttempts++;
      this.lastTDError = meanTDError;

      if (meanTDError >= TD_ERROR_THRESHOLD) {
        // Treina
        await this.model.fit(stateTensor, targets, {
          epochs: 1,
          batchSize: BATCH_SIZE,
          verbose: 0
        });

        this.decayEpsilon();
        this.totalTrained++;
        this.batchesSinceLastTrain = 0;
      } else {
        // Pula
        this.batchesSinceLastTrain++;
      }

      targets.dispose();

    } catch (error) {
      console.error('Training error:', error);
    } finally {
      stateTensor.dispose();
      nextStateTensor.dispose();
      rewardTensor.dispose();
      doneTensor.dispose();
      this.trainingInProgress = false;
    }

    // Atualiza target network
    if (this.stepCount % TARGET_UPDATE_FREQ === 0) {
      this.targetModel.setWeights(this.model.getWeights());
    }
  }

  getTrainStats() {
    const attempts = this.totalTrainAttempts || 0;
    const trained = this.totalTrained || 0;
    const pct = attempts > 0 ? (trained / attempts) * 100 : 0;

    return {
      trained,
      attempts,
      pct: pct.toFixed(1),
      lastTDError: this.lastTDError ? this.lastTDError.toFixed(3) : '—',
      threshold: TD_ERROR_THRESHOLD.toFixed(3)
    };
  }

  async saveBrain(generation, highScore = 0) {
    await this.persistence.saveModel(this.model);
    await this.persistence.saveReplayBuffers(this.replayBuffer);
    await this.persistence.saveMetadata({ epsilon, generation, highScore });
    console.log('Estado completo salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      const model = await this.persistence.loadModel();
      if (model) {
        const modelInputSize = model.inputs[0].shape[1];
        if (modelInputSize !== STATE_SIZE) {
          console.log(`Modelo salvo tem input ${modelInputSize}, esperado ${STATE_SIZE}. Resetando cérebro antigo...`);
          await this.persistence.clearAll();
          return { success: false, generation: 1 };
        }

        this.model = model;
        this.model.compile({
          optimizer: tf.train.adam(LEARNING_RATE),
          loss: 'meanSquaredError'
        });

        this.targetModel = this.createModel();
        this.targetModel.setWeights(model.getWeights());

        await this.persistence.loadReplayBuffers(this.replayBuffer);

        const metadata = await this.persistence.loadMetadata();
        if (metadata) {
          epsilon = metadata.epsilon ?? 0.9;
          const gen = metadata.generation ?? 1;
          const highScore = metadata.highScore ?? 0;
          console.log('Estado completo carregado! Epsilon:', epsilon, 'Gen:', gen, 'HighScore:', highScore);
          return { success: true, generation: gen, highScore };
        }
      }
    } catch (e) {
      console.log('Nenhum estado salvo encontrado');
    }

    return { success: false, generation: 1 };
  }

  decayEpsilon() {
    if (epsilon > EPSILON_MIN) {
      epsilon *= EPSILON_DECAY;
      if (epsilon < EPSILON_MIN) epsilon = EPSILON_MIN;
    }
  }

  getQValues(state) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
      const qValues = this.model.predict(stateTensor);
      const qArray = qValues.dataSync();
      return {
        [ACTION_IDLE]: qArray[0],
        [ACTION_FLAP]: qArray[1]
      };
    });
  }
}

export async function resetBrain() {
  const pm = new PersistenceManager();
  await pm.clearAll();
  epsilon = 0.9;
  console.log('Jogo completamente resetado');
}