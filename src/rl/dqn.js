/* ============================================================
 * DEEP Q-NETWORK — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Uses TensorFlow.js to implement DQN with a neural network.
 * States are continuous [dx, dy, velY], no discretization.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { ReplayBuffer } from './replayBuffer.js';  // ← IMPORT AQUI (ajuste path se necessário)

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
    this.replayBuffer = new ReplayBuffer();  // ← Usa defaults (50k reservoir + 10k recent)
    this.stepCount = 0;
    this.trainingInProgress = false; // Flag para evitar treinos concorrentes
  }

  createModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({ units: 512, activation: 'relu', inputShape: [8] }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'linear' }));

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
    // CORREÇÃO: Incrementa stepCount SEMPRE, no início, para evitar congelamento
    this.stepCount++;

    // Throttle: Treina só a cada TRAIN_THROTTLE passos (agora stepCount já avançou)
    if (this.stepCount % TRAIN_THROTTLE !== 0) {
      // Opcional: Log para debug (comente se quiser silenciar)
      // console.log('Step skipped:', this.stepCount);
      return;
    }
    if (this.replayBuffer.size() < BATCH_SIZE) return;
    if (this.trainingInProgress) {
      console.log('Training skipped: Already in progress');
      return;
    }

    this.trainingInProgress = true;

    // Usa o híbrido (reservoir + recent real)
    const batch = this.replayBuffer.sampleLastPrioritizedAndRandom(BATCH_SIZE);  // Ou passe %: sampleHybrid(BATCH_SIZE, 0.4)

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

    // Create input tensors
    const stateTensor = tf.tensor2d(states, [BATCH_SIZE, 8]);
    const nextStateTensor = tf.tensor2d(nextStates, [BATCH_SIZE, 8]);
    const rewardTensor = tf.tensor1d(rewards);
    const doneTensor = tf.tensor1d(dones);

    try {
      // Compute targets inside tidy
      const targets = tf.tidy(() => {
        const nextQValues = this.targetModel.predict(nextStateTensor);
        const maxNextQ = nextQValues.max(1);
        const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
        const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));
        const qValues = this.model.predict(stateTensor);
        const qValuesArray = qValues.arraySync();
        const targetQArray = targetQ.arraySync();
        for (let i = 0; i < BATCH_SIZE; i++) {
          qValuesArray[i][actions[i]] = targetQArray[i];
        }
        const targets = tf.tensor2d(qValuesArray, [BATCH_SIZE, 2]);
        nextQValues.dispose();
        maxNextQ.dispose();
        targetQ.dispose();
        qValues.dispose();
        return targets;
      });

      // Fit outside tidy
      await this.model.fit(stateTensor, targets, {
        epochs: 1,
        batchSize: BATCH_SIZE,
        verbose: 0
      });

      this.decayEpsilon();

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

    // CORREÇÃO: Check de target update após incremento (agora sempre executa)
    if (this.stepCount % TARGET_UPDATE_FREQ === 0) {
      this.targetModel.setWeights(this.model.getWeights());
    }
  }

  async saveBrain(generation) {
    await this.model.save('localstorage://flappy-dqn');
    localStorage.setItem('flappy_dqn_metadata', JSON.stringify({
      epsilon,
      generation
    }));
    console.log('DQN Brain saved! Gen:', generation);
  }

  async loadBrain() {
    try {
      const model = await tf.loadLayersModel('localstorage://flappy-dqn');
      this.model = model;
      // CORREÇÃO: Recompile o modelo carregado para restaurar optimizer/loss
      this.model.compile({
        optimizer: tf.train.adam(LEARNING_RATE),
        loss: 'meanSquaredError'
      });
      this.targetModel = this.createModel();
      this.targetModel.setWeights(model.getWeights());
      const metadata = JSON.parse(localStorage.getItem('flappy_dqn_metadata'));
      if (metadata) {
        epsilon = metadata.epsilon;
        const generation = metadata.generation;
        console.log('DQN Brain loaded! Epsilon:', epsilon, 'Gen:', generation);
        return { success: true, generation };
      }
    } catch (e) {
      console.log('No DQN brain found');
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

export function resetBrain() {
  localStorage.removeItem('flappy-dqn');
  localStorage.removeItem('flappy_dqn_metadata');
  epsilon = 0.1;
  console.log('DQN Brain reset');
}