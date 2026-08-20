/* ============================================================
 * DEEP Q-NETWORK — FLAPPY BIRD ADAPTER
 * ------------------------------------------------------------
 * Uses TensorFlow.js to implement DQN with a neural network.
 * States are continuous [dx, dy, velY], no discretization.
 * ============================================================ */

import * as tf from '@tensorflow/tfjs';
import { ReplayBuffer } from './replayBuffer.js';  // ← IMPORT AQUI (ajuste path se necessário)
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
export const gamma = 0.99; // Discount Factor
export let epsilon = 0.9; // Exploration Rate
export const EPSILON_MIN = 0.000;
export const EPSILON_DECAY = 0.9995;
export const BATCH_SIZE = 64;
export const TARGET_UPDATE_FREQ = 1000; // Atualiza target network com menos frequência
export const LEARNING_RATE = 0.001; // Learning rate menor para convergência suave
export const TRAIN_THROTTLE = 2; // Treina a cada N passos para evitar sobrecarga

/* ------------------------------------------------------------
 * NETWORK EXPANSION (Forget-and-Grow) — Hiperparámetros
 * ------------------------------------------------------------ */
export const INITIAL_HIDDEN_UNITS = 64;      // unidades ocultas iniciais
export const EXPAND_EVERY_GENERATIONS = 80;  // expandir cada N gerações
export const EXPAND_UNITS = 16;              // unidades adicionadas por expansão
export const MAX_HIDDEN_UNITS = 192;         // limite superior

/* ------------------------------------------------------------
 * DQN AGENT
 * ------------------------------------------------------------ */
export class DQNAgent {
  constructor() {

    // Force WebGL backend for better performance and stability
    tf.setBackend('webgl').then(() => console.log('TensorFlow.js backend: WebGL')).catch(err => console.warn('Failed to set WebGL:', err));
    this.hiddenUnits = INITIAL_HIDDEN_UNITS;
    this.expansionCount = 0;
    this.model = this.createModel();
    this.targetModel = this.createModel();
    this.targetModel.setWeights(this.model.getWeights());
    this.replayBuffer = new ReplayBuffer();  // ← Usa defaults (50k reservoir + 10k recent)
    this.stepCount = 0;
    this.trainingInProgress = false; // Flag para evitar treinos concorrentes

    // Load model from IndexedDB or create new
    this.persistence = new PersistenceManager();
  }

  createModel(hiddenUnits = this.hiddenUnits) {
    const model = tf.sequential();

    // Input de STATE_SIZE para hiddenUnits neurônios
    model.add(tf.layers.dense({ units: hiddenUnits, activation: 'relu', inputShape: [STATE_SIZE] }));
    
    // hiddenUnits neurônios
    model.add(tf.layers.dense({ units: hiddenUnits, activation: 'relu' }));
    
    // Camada de saída (2 ações)
    model.add(tf.layers.dense({ units: 2, activation: 'linear' }));

    model.compile({
        optimizer: tf.train.adam(LEARNING_RATE),
        loss: 'meanSquaredError'
    });

    return model;
  }

  // ------------------------------------------------------------
  // NETWORK EXPANSION (Forget-and-Grow) — Neurogênese
  // ------------------------------------------------------------
  expandNetwork(extraUnits = EXPAND_UNITS) {
    const newUnits = this.hiddenUnits + extraUnits;
    if (newUnits > MAX_HIDDEN_UNITS) {
      console.log(`[FoG] Red no máximo ${MAX_HIDDEN_UNITS} unidades ocultas. Expansão omitida.`);
      return false;
    }

    const oldUnits = this.hiddenUnits;
    const oldWeights = this.model.getWeights();
    const newModel = this.createModel(newUnits);

    // Copia os pesos antigos e adiciona unidades novas com inicialização aleatoria (Xavier)
    tf.tidy(() => {
      const w1 = oldWeights[0]; // [STATE_SIZE, oldUnits]
      const b1 = oldWeights[1]; // [oldUnits]
      const w2 = oldWeights[2]; // [oldUnits, oldUnits]
      const b2 = oldWeights[3]; // [oldUnits]
      const w3 = oldWeights[4]; // [oldUnits, 2]
      const b3 = oldWeights[5]; // [2]

      // Capa 1: input → hidden (só cresce em colunas)
      const newW1 = tf.concat([w1, tf.randomNormal([STATE_SIZE, extraUnits])], 1);
      const newB1 = tf.concat([b1, tf.randomNormal([extraUnits])], 0);

      // Capa 2: hidden → hidden (cresce em filas e colunas)
      const topLeft = w2;
      const topRight = tf.randomNormal([oldUnits, extraUnits]);
      const bottomLeft = tf.randomNormal([extraUnits, oldUnits]);
      const bottomRight = tf.randomNormal([extraUnits, extraUnits]);
      const newW2 = tf.concat(
        [tf.concat([topLeft, topRight], 1), tf.concat([bottomLeft, bottomRight], 1)],
        0
      );
      const newB2 = tf.concat([b2, tf.randomNormal([extraUnits])], 0);

      // Capa 3: hidden → output (só cresce em filas)
      const newW3 = tf.concat([w3, tf.randomNormal([extraUnits, 2])], 0);
      const newB3 = b3;

      newModel.setWeights([newW1, newB1, newW2, newB2, newW3, newB3]);
    });

    this.model = newModel;
    this.model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
    this.targetModel = this.createModel(newUnits);
    this.targetModel.setWeights(this.model.getWeights());
    this.hiddenUnits = newUnits;
    this.expansionCount++;

    // Tras a expansão, aumenta ligeramente a exploração para testar as novas unidades
    epsilon = Math.min(epsilon + 0.05, 0.9);

    console.log(`[FoG] Red expandida: ${oldUnits} → ${newUnits} unidades ocultas (expansão #${this.expansionCount})`);
    return true;
  }

  maybeExpandNetwork(generation) {
    if (generation > 0 && generation % EXPAND_EVERY_GENERATIONS === 0) {
      return this.expandNetwork();
    }
    return false;
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

    // FoG (ER Decay): amostragem híbrida com decaimento temporal no reservoir
    const batch = this.replayBuffer.sampleHybridWithDecay(BATCH_SIZE);

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
    const stateTensor = tf.tensor2d(states, [BATCH_SIZE, STATE_SIZE]);
    const nextStateTensor = tf.tensor2d(nextStates, [BATCH_SIZE, STATE_SIZE]);
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

  async saveBrain(generation, highScore = 0) {
    await this.persistence.saveModel(this.model);
    await this.persistence.saveReplayBuffers(this.replayBuffer);
    await this.persistence.saveMetadata({ epsilon, generation, highScore, hiddenUnits: this.hiddenUnits, expansionCount: this.expansionCount });
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
        this.model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
        // Detecta as unidades ocultas a partir do modelo cargado (arquitetura variável)
        this.hiddenUnits = (model.layers[0] && model.layers[0].units) ? model.layers[0].units : INITIAL_HIDDEN_UNITS;
        this.targetModel = this.createModel(this.hiddenUnits);
        this.targetModel.setWeights(model.getWeights());

        await this.persistence.loadReplayBuffers(this.replayBuffer);

        const metadata = await this.persistence.loadMetadata();
        if (metadata) {
          epsilon = metadata.epsilon ?? 0.9;
          this.hiddenUnits = metadata.hiddenUnits ?? this.hiddenUnits;
          this.expansionCount = metadata.expansionCount ?? 0;
          const gen = metadata.generation ?? 1;
          const highScore = metadata.highScore ?? 0;
          console.log('Estado completo carregado! Epsilon:', epsilon, 'Gen:', gen, 'HiddenUnits:', this.hiddenUnits, 'HighScore:', highScore);
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