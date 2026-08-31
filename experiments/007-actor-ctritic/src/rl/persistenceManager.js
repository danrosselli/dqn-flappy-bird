// persistenceManager.js
// ------------------------------------------------------------
// Persistence for the Policy Gradient experiment.
// Only the neural-network model and experiment metadata are kept.
// Replay buffers are intentionally not persisted because REINFORCE
// uses a temporary trajectory for each episode.
// ------------------------------------------------------------

import * as tf from '@tensorflow/tfjs';
import experimentConfig from '../../experiment.json';

const EXPERIMENT_NAME =
  (experimentConfig && experimentConfig.name)
    ? experimentConfig.name
    : 'experiment';

const DB_NAME = `FlappyPolicyGradientDB_${EXPERIMENT_NAME}`;
const DB_VERSION = 1;
const GAME_DATA_STORE = 'gameData';

export class PersistenceManager {
  constructor() {
    this.modelPath =
      `indexeddb://flappy-policy-gradient-model-${EXPERIMENT_NAME}`;
  }

  async openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains(GAME_DATA_STORE)) {
          db.createObjectStore(GAME_DATA_STORE);
        }
      };
    });
  }

  async saveModel(model) {
    try {
      await model.save(this.modelPath);
      console.log('Modelo Policy Gradient salvo no IndexedDB');
    } catch (error) {
      console.error('Erro ao salvar modelo:', error);
    }
  }

  async loadModel() {
    try {
      const model = await tf.loadLayersModel(this.modelPath);
      console.log('Modelo Policy Gradient carregado do IndexedDB');
      return model;
    } catch (error) {
      console.log('Nenhum modelo Policy Gradient encontrado');
      return null;
    }
  }

  async deleteModel() {
    try {
      await tf.io.removeModel(this.modelPath);
    } catch (error) {
      // removeModel throws when there is nothing to remove.
      console.warn('Modelo Policy Gradient inexistente ou já removido');
    }
  }

  async saveGameData(key, data) {
    const db = await this.openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(GAME_DATA_STORE, 'readwrite');
      const store = transaction.objectStore(GAME_DATA_STORE);

      transaction.oncomplete = () => {
        db.close();
        resolve();
      };

      transaction.onerror = () => {
        db.close();
        reject(transaction.error);
      };

      store.put(data, key);
    });
  }

  async loadGameData(key) {
    const db = await this.openDB();

    return new Promise((resolve) => {
      const transaction = db.transaction(GAME_DATA_STORE, 'readonly');
      const store = transaction.objectStore(GAME_DATA_STORE);
      const request = store.get(key);

      request.onsuccess = () => {
        db.close();
        resolve(request.result ?? null);
      };

      request.onerror = () => {
        db.close();
        resolve(null);
      };
    });
  }

  async deleteGameData(key) {
    const db = await this.openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(GAME_DATA_STORE, 'readwrite');
      const store = transaction.objectStore(GAME_DATA_STORE);

      transaction.oncomplete = () => {
        db.close();
        resolve();
      };

      transaction.onerror = () => {
        db.close();
        reject(transaction.error);
      };

      store.delete(key);
    });
  }

  async saveMetadata(metadata) {
    await this.saveGameData('metadata', metadata);
  }

  async loadMetadata() {
    return this.loadGameData('metadata');
  }

  async clearAll() {
    await this.deleteModel();

    try {
      await this.deleteGameData('metadata');
    } catch (error) {
      console.warn('Não foi possível remover metadata:', error);
    }

    console.log('Memória do Policy Gradient resetada');
  }
}
