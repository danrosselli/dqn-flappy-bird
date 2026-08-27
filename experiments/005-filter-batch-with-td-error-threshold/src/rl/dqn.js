/* ============================================================
 * DEEP Q-NETWORK — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Uses TensorFlow.js to implement DQN with a neural network.
 * States are continuous [dx, dy, velY], no discretization.
 *
 * Adaptive TD-error filtering:
 * - Samples a variable number of transitions
 * - Keeps only those with normalized TD-error >= threshold
 * - Adjusts next sample size based on pass rate
 * - Throttled training (every 2 steps)
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
export const TARGET_UPDATE_FREQ = 1000;
export const LEARNING_RATE = 0.001;
export const TRAIN_THROTTLE = 2;

// --- Adaptive Prioritized Training / TD-Error Filter ---
export const TD_NORM_THRESHOLD = 0.04;
export const INITIAL_SAMPLE_SIZE = 32;
export const MIN_VALID_SAMPLES = 12;
export const MAX_SAMPLE_SIZE = 128;

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

    // Adaptive sampling state
    this.sampleSize = INITIAL_SAMPLE_SIZE;
    this.batchesSinceLastTrain = 0;
    this.lastTDError = 0;
    this.lastValidCount = 0;
    this.lastSampleSize = INITIAL_SAMPLE_SIZE;

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

    // Throttle: treina a cada 2 passos
    if (this.stepCount % TRAIN_THROTTLE !== 0) {
      return;
    }

    // Precisa ter pelo menos o tamanho atual de amostragem no buffer
    if (this.replayBuffer.size() < this.sampleSize) return;

    if (this.trainingInProgress) {
      console.log('Training skipped: Already in progress');
      return;
    }

    this.trainingInProgress = true;
    this.totalTrainAttempts++;

    // Amostra o batch com tamanho adaptativo
    const batch = this.replayBuffer.sampleRandomBasic(this.sampleSize);
    const currentSampleSize = batch.length;

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

    const stateTensor = tf.tensor2d(states, [currentSampleSize, STATE_SIZE]);
    const nextStateTensor = tf.tensor2d(nextStates, [currentSampleSize, STATE_SIZE]);
    const rewardTensor = tf.tensor1d(rewards);
    const doneTensor = tf.tensor1d(dones);

    try {
      // 1. Calcula targets + filtra por TD-error normalizado individual
      const {
        validStates,
        validTargets,
        meanTDError,
        meanNormTDError,
        validCount
      } = tf.tidy(() => {
        const nextQValues = this.targetModel.predict(nextStateTensor);
        const maxNextQ = nextQValues.max(1);
        const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
        const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));

        const qValues = this.model.predict(stateTensor);
        const qValuesArray = qValues.arraySync();
        const targetQArray = targetQ.arraySync();

        const validStatesArr = [];
        const validTargetsArr = [];
        let totalAbsError = 0;
        let totalNormError = 0;

        for (let i = 0; i < currentSampleSize; i++) {
          const currentQ = qValuesArray[i][actions[i]];
          const target = targetQArray[i];

          const absError = Math.abs(target - currentQ);
          const normError = absError / (Math.abs(target) + 1e-6);

          totalAbsError += absError;
          totalNormError += normError;

          // Mantém apenas amostras com TD-error normalizado relevante
          if (normError >= TD_NORM_THRESHOLD) {
            const sampleTarget = qValuesArray[i].slice(); // copia [Q_idle, Q_flap]
            sampleTarget[actions[i]] = target;

            validStatesArr.push(states[i]);
            validTargetsArr.push(sampleTarget);
          }
        }

        const meanTDError = totalAbsError / currentSampleSize;
        const meanNormTDError = totalNormError / currentSampleSize;

        nextQValues.dispose();
        maxNextQ.dispose();
        targetQ.dispose();
        qValues.dispose();

        return {
          validStates: validStatesArr,
          validTargets: validTargetsArr,
          meanTDError,
          meanNormTDError,
          validCount: validStatesArr.length
        };
      });

      this.lastTDError = meanNormTDError;
      this.lastValidCount = validCount;
      this.lastSampleSize = currentSampleSize;

      // 2. Treina somente se sobrou um número mínimo de amostras válidas
      if (validCount >= MIN_VALID_SAMPLES) {
        const validStateTensor = tf.tensor2d(validStates, [validCount, STATE_SIZE]);
        const targetsTensor = tf.tensor2d(validTargets, [validCount, 2]);

        await this.model.fit(validStateTensor, targetsTensor, {
          epochs: 1,
          batchSize: validCount,
          verbose: 0
        });

        this.decayEpsilon();
        this.totalTrained++;
        this.batchesSinceLastTrain = 0;

        validStateTensor.dispose();
        targetsTensor.dispose();
      } else {
        this.batchesSinceLastTrain++;
      }

      // 3. Ajusta o tamanho de amostragem para a próxima vez
      this._adjustSampleSize(validCount, currentSampleSize);

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

  /**
   * Ajusta this.sampleSize com base na taxa de sobrevivência do filtro.
   */
  _adjustSampleSize(validCount, sampledCount) {
    if (sampledCount === 0) return;

    const passRate = validCount / sampledCount;

    if (validCount < MIN_VALID_SAMPLES) {
      // Quase nada passou → aumenta bastante
      this.sampleSize = Math.min(
        Math.ceil(this.sampleSize * 2),
        MAX_SAMPLE_SIZE
      );
    } else if (passRate < 0.4) {
      // Filtro rejeitando muito → aumenta amostragem
      this.sampleSize = Math.min(
        Math.ceil(this.sampleSize * 1.5),
        MAX_SAMPLE_SIZE
      );
    } else if (passRate > 0.8 && this.sampleSize > INITIAL_SAMPLE_SIZE) {
      // Quase tudo passa → pode diminuir um pouco
      this.sampleSize = Math.max(
        INITIAL_SAMPLE_SIZE,
        Math.floor(this.sampleSize * 0.8)
      );
    }
    // Caso intermediário (0.4 ~ 0.8): mantém o tamanho atual
  }

  getTrainStats() {
    const attempts = this.totalTrainAttempts || 0;
    const trained = this.totalTrained || 0;

    // Taxa de aproveitamento da última amostragem
    const passRate = this.lastSampleSize > 0
      ? (this.lastValidCount / this.lastSampleSize) * 100
      : 0;

    return {
      trained,
      attempts,
      pct: passRate.toFixed(1),          // agora representa % de amostras que passaram no filtro
      lastTDError: this.lastTDError ? this.lastTDError.toFixed(3) : '—',
      threshold: TD_NORM_THRESHOLD.toFixed(3),
      sampleSize: this.sampleSize,
      lastValid: this.lastValidCount,
      lastSampled: this.lastSampleSize
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