/* ============================================================
 * DEEP Q-NETWORK — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Uses TensorFlow.js to implement DQN with a neural network.
 * States are continuous [dx, dy, velY], no discretization.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
//import { ReplayBuffer } from './replayBuffer.js';  // ← IMPORT AQUI (ajuste path se necessário)
import { PERReplayBuffer } from './perReplayBuffer.js';  // Ajuste o path se necessário
import { PersistenceManager } from './persistenceManager.js';

/* ------------------------------------------------------------
 * ACTIONS
 * ------------------------------------------------------------ */
export const ACTION_FLAP = 1;
export const ACTION_IDLE = 0;
export const ACTIONS = [ACTION_IDLE, ACTION_FLAP];

/* ------------------------------------------------------------
 * HYPERPARAMETERS
 * ------------------------------------------------------------ */
export const gamma = 0.99; // Discount Factor
export let epsilon = 0.9; // Exploration Rate
export const EPSILON_MIN = 0.000;
export const EPSILON_DECAY = 0.9995;
export const BATCH_SIZE = 64;
export const TARGET_UPDATE_FREQ = 1000; // Atualiza target network com menos frequência
export const LEARNING_RATE = 0.001; // Learning rate menor para convergência suave
export const TRAIN_THROTTLE = 2; // Treina a cada N passos para evitar sobrecarga

// Classe custom com versão soft (diferenciável)
class SparseReLU extends tf.layers.Layer {
  constructor(config = {}) {
    super(config);
    this.threshold = config.threshold || 0.1;   // Threshold para esparsidade
    this.steepness = config.steepness || 50;    // Quanto maior, mais "hard" (aproxima threshold rígido)
  }

  call(inputs) {
    return tf.tidy(() => {
      const relu = tf.relu(inputs);
      const abs = tf.abs(inputs);
      // Soft mask: sigmoid íngreme para aproximar hard threshold
      const diff = tf.sub(abs, this.threshold);
      const softMask = tf.sigmoid(tf.mul(this.steepness, diff));
      return tf.mul(relu, softMask);
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    return {
      threshold: this.threshold,
      steepness: this.steepness
    };
  }

  static get className() {
    return 'SparseReLU';
  }
}

// Registre (uma vez só)
tf.serialization.registerClass(SparseReLU);


/* ------------------------------------------------------------
 * DQN AGENT
 * ------------------------------------------------------------ */
export class DQNAgent {
  constructor() {

    // Force WebGL backend for better performance and stability
    tf.setBackend('webgl').then(() => console.log('TensorFlow.js backend: WebGL')).catch(err => console.warn('Failed to set WebGL:', err));
    this.model = this.createModel();
    this.targetModel = this.createModel();
    this.targetModel.setWeights(this.model.getWeights());
    this.replayBuffer = new PERReplayBuffer();  // ← Usa defaults (50k reservoir + 10k recent)
    this.stepCount = 0;
    this.trainingInProgress = false; // Flag para evitar treinos concorrentes

    this.beta = 0.4; // Inicialização beta para PER
    this.betaIncrement = (1.0 - 0.4) / 100000; // Aumenta gradualmente para 1.0 (ajuste o denominador baseado nas suas steps/gerações totais)

    // Load model from IndexedDB or create new
    this.persistence = new PersistenceManager();
  }

  createModel() {
    const threshold = 0.1;    // Teste 0.05 (menos esparsidade) a 0.2 (mais)
    const steepness = 50;     // Teste 20 (mais soft) a 100 (quase hard)

    const model = tf.sequential();

    model.add(tf.layers.dense({
      units: 512,
      activation: { className: 'SparseReLU', config: { threshold, steepness } },
      inputShape: [8]
    }));

    model.add(tf.layers.dense({
      units: 256,
      activation: { className: 'SparseReLU', config: { threshold, steepness } }
    }));

    model.add(tf.layers.dense({
      units: 128,
      activation: { className: 'SparseReLU', config: { threshold, steepness } }
    }));

    model.add(tf.layers.dense({
      units: 64,
      activation: { className: 'SparseReLU', config: { threshold, steepness } }
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
      const stateTensor = tf.tensor2d([state], [1, 8]);
      const qValues = this.model.predict(stateTensor);
      const action = qValues.argMax(1).dataSync()[0];
      return action;
    });
  }

  async train() {
    this.stepCount++;

    if (this.stepCount % TRAIN_THROTTLE !== 0) {
      return;
    }
    if (this.replayBuffer.size() < BATCH_SIZE) return;
    if (this.trainingInProgress) {
      console.log('Training skipped: Already in progress');
      return;
    }

    this.trainingInProgress = true;

    // Sample com PER (retorna {batch, indices, weights})
    const { batch, indices, weights } = this.replayBuffer.sample(BATCH_SIZE, this.beta);

    if (batch.length === 0) {  // Caso raro: buffer vazio ou erro
      this.trainingInProgress = false;
      return;
    }

    const states = batch.map(t => t.state);
    const actions = batch.map(t => t.action);
    const rewards = batch.map(t => t.reward);
    const nextStates = batch.map(t => t.nextState);
    const dones = batch.map(t => t.done ? 1 : 0);

    const stateTensor = tf.tensor2d(states, [batch.length, 8]);
    const nextStateTensor = tf.tensor2d(nextStates, [batch.length, 8]);
    const rewardTensor = tf.tensor1d(rewards);
    const doneTensor = tf.tensor1d(dones);
    const sampleWeightsTensor = tf.tensor1d(weights);
    const actionsTensor = tf.tensor1d(actions, 'int32');

    try {
      // 1. Calcula os alvos (Target Q) e TD Errors para o buffer
      const { targetQ, tdErrorsArray } = tf.tidy(() => {
        const nextQValues = this.targetModel.predict(nextStateTensor);
        const maxNextQ = nextQValues.max(1);
        const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
        const tq = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));

        const qValues = this.model.predict(stateTensor);
        const oneHotActions = tf.oneHot(actionsTensor, 2);
        const selectedQ = tf.sum(qValues.mul(oneHotActions), 1);
        const tdErrors = tq.sub(selectedQ).abs();

        return {
          targetQ: tq,
          tdErrorsArray: tdErrors.dataSync()
        };
      });

      // 2. Fit manual com pesos (Importance Sampling)
      this.model.optimizer.minimize(() => {
        const qValues = this.model.predict(stateTensor);
        const oneHotActions = tf.oneHot(actionsTensor, 2);
        const selectedQ = tf.sum(qValues.mul(oneHotActions), 1);

        // Weighted Mean Squared Error
        const diff = tf.sub(targetQ, selectedQ);
        const loss = tf.mean(tf.mul(tf.square(diff), sampleWeightsTensor));
        return loss;
      });

      // Atualiza priorities no buffer
      this.replayBuffer.updatePriorities(indices, tdErrorsArray);

      // Annealing beta (aumenta gradualmente para 1.0)
      this.beta = Math.min(1.0, this.beta + this.betaIncrement);

      this.decayEpsilon();

      targetQ.dispose();
    } catch (error) {
      console.error('Training error:', error);
    } finally {
      stateTensor.dispose();
      nextStateTensor.dispose();
      rewardTensor.dispose();
      doneTensor.dispose();
      sampleWeightsTensor.dispose();
      actionsTensor.dispose();
      this.trainingInProgress = false;
    }

    if (this.stepCount % TARGET_UPDATE_FREQ === 0) {
      this.targetModel.setWeights(this.model.getWeights());
    }
  }

  async saveBrain(generation) {
    await this.persistence.saveModel(this.model);
    await this.persistence.saveReplayBuffers(this.replayBuffer);
    await this.persistence.saveMetadata({ epsilon, generation });
    console.log('Estado completo salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      const model = await this.persistence.loadModel();
      if (model) {
        this.model = model;
        this.model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
        this.targetModel = this.createModel();
        this.targetModel.setWeights(model.getWeights());

        await this.persistence.loadReplayBuffers(this.replayBuffer);

        const metadata = await this.persistence.loadMetadata();
        if (metadata) {
          epsilon = metadata.epsilon ?? 0.9;
          const gen = metadata.generation ?? 1;
          console.log('Estado completo carregado! Epsilon:', epsilon, 'Gen:', gen);
          return { success: true, generation: gen };
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
      const stateTensor = tf.tensor2d([state], [1, 8]);
      const qValues = this.model.predict(stateTensor);
      const qArray = qValues.dataSync();
      return { [ACTION_IDLE]: qArray[0], [ACTION_FLAP]: qArray[1] };
    });
  }
}

export async function resetBrain() {
  const pm = new PersistenceManager();
  await pm.clearAll();
  epsilon = 0.9;
  console.log('Jogo completamente resetado');
}