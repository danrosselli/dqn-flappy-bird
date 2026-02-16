/* ============================================================
 * DEEP Q-NETWORK — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Uses TensorFlow.js to implement DQN with a neural network.
 * States are continuous [dx, dy, velY], no discretization.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { NestedModel } from './NestedModel.js';
import { ReplayBuffer } from './replayBuffer.js';  // ← IMPORT AQUI (ajuste path se necessário)
import { PERReplayBuffer } from './perReplayBuffer.js';  // Ajuste o path se necessário
import { StratifiedPER } from './stratifiedPER.js';  // Ajuste o path se necessário
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
export const TARGET_UPDATE_FREQ = 2000; // Atualiza target network com menos frequência
export const LEARNING_RATE = 0.001; // Learning rate menor para convergência suave
export const TRAIN_THROTTLE = 4; // Treina a cada N passos para evitar sobrecarga

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
    this.model = new NestedModel();
    this.targetModel = new NestedModel();
    this.targetModel.syncFrom(this.model);
    this.replayBuffer = new StratifiedPER(15000, 0.6); // 15k por estrato = 45k total
    this.stepCount = 0;
    this.trainingInProgress = false; // Flag para evitar treinos concorrentes

    this.beta = 0.4; // Inicialização beta para PER
    this.betaIncrement = (1.0 - 0.4) / 100000; // Aumenta gradualmente para 1.0 (ajuste o denominador baseado nas suas steps/gerações totais)

    // Load model from IndexedDB or create new
    this.persistence = new PersistenceManager();

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
    if (this.stepCount % TRAIN_THROTTLE !== 0) return;
    if (this.replayBuffer.size() < BATCH_SIZE) return;
    if (this.trainingInProgress) return;
    this.trainingInProgress = true;

    // === Sampling unificado: sempre retorna batch ===
    const isPER = this.replayBuffer.constructor.name === 'StratifiedPER';
    let batch;
    if (isPER) {
      batch = this.replayBuffer.sample(BATCH_SIZE, this.beta);
    } else {
      // Para buffer simples: use o método que deu alto score
      batch = this.replayBuffer.sampleRandomBasic(BATCH_SIZE);
      // Adiciona weight=1.0 em todas (para compatibilidade)
      batch.forEach(t => t.weight = 1.0);
    }

    if (batch.length === 0) {
      this.trainingInProgress = false;
      return;
    }

    // Extrai dados
    const states = batch.map(t => t.state);
    const actions = batch.map(t => t.action);
    const rewards = batch.map(t => t.reward);
    const nextStates = batch.map(t => t.nextState);
    const dones = batch.map(t => t.done ? 1 : 0);
    const weights = batch.map(t => t.weight || 1.0);  // Usa .weight se PER, senão 1

    // Tensores...
    const stateTensor = tf.tensor2d(states, [batch.length, 8]);
    const nextStateTensor = tf.tensor2d(nextStates, [batch.length, 8]);
    const rewardTensor = tf.tensor1d(rewards);
    const doneTensor = tf.tensor1d(dones);
    const sampleWeightsTensor = tf.tensor1d(weights);
    const actionsTensor = tf.tensor1d(actions, 'int32');

    try {
      const { targetQ, tdErrorsArray } = tf.tidy(() => {
        // === DDQN (igual) ===
        const nextQOnline = this.model.predict(nextStateTensor);
        const nextActions = nextQOnline.argMax(1);
        const nextQTarget = this.targetModel.predict(nextStateTensor);
        const oneHotNext = tf.oneHot(nextActions, 2);
        const maxNextQ = tf.sum(nextQTarget.mul(oneHotNext), 1);
        const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
        const tq = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));

        const qValues = this.model.predict(stateTensor);
        const oneHotActions = tf.oneHot(actionsTensor, 2);
        const selectedQ = tf.sum(qValues.mul(oneHotActions), 1);
        const tdErrors = tq.sub(selectedQ).abs();

        nextQOnline.dispose();
        nextQTarget.dispose();
        oneHotNext.dispose();

        return { targetQ: tq, tdErrorsArray: tdErrors.dataSync() };
      });

      // CERTO - usar trainStep do NestedModel:
      await this.model.trainStep(stateTensor, targetQ, actionsTensor, sampleWeightsTensor);

      // === Adiciona tdError nas transitions (para PER) ===
      batch.forEach((t, i) => t.tdError = tdErrorsArray[i]);

      // === Update priorities só se PER ===
      if (isPER) {
        this.replayBuffer.updatePriorities(batch);
        this.beta = Math.min(1.0, this.beta + this.betaIncrement);
      }

      this.decayEpsilon();
      targetQ.dispose();

    } catch (error) {
      console.error('Training error:', error);
    } finally {
      // Dispose tensores...
      stateTensor.dispose();
      nextStateTensor.dispose();
      rewardTensor.dispose();
      doneTensor.dispose();
      sampleWeightsTensor.dispose();
      actionsTensor.dispose();
      this.trainingInProgress = false;
    }

    if (this.stepCount % TARGET_UPDATE_FREQ === 0) {
      this.targetModel.syncFrom(this.model);
    }
  }

  async saveBrain(generation) {
    // Salva NestedModel serializado (novo formato)
    const modelData = this.model.serialize();
    await this.persistence.saveNestedModel(modelData);

    // Salva buffers e metadata
    await this.persistence.saveReplayBuffers(this.replayBuffer);
    await this.persistence.saveMetadata({
      epsilon,
      generation,
      stepCount: this.stepCount
    });

    console.log('Estado completo salvo! Gen:', generation);
  }

  async loadBrain() {
    try {
      // Tenta carregar NestedModel primeiro
      const modelData = await this.persistence.loadNestedModel();
      if (modelData) {
        await this.model.deserialize(modelData);
        this.targetModel.syncFrom(this.model); // Sincroniza target

        await this.persistence.loadReplayBuffers(this.replayBuffer);
        const metadata = await this.persistence.loadMetadata();

        if (metadata) {
          epsilon = metadata.epsilon ?? 0.9;
          this.stepCount = metadata.stepCount || 0;
          const gen = metadata.generation ?? 1;
          console.log('NestedModel carregado! Gen:', gen, 'Step:', this.stepCount);
          return { success: true, generation: gen };
        }
      }

      // Fallback: tenta carregar modelo TF.js legado (para compatibilidade)
      // ... código legado se necessário ...

    } catch (e) {
      console.log('Nenhum estado salvo encontrado:', e.message);
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