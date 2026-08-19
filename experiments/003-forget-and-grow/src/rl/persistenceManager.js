// persistenceManager.js
// ------------------------------------------------------------
// Classe única para gerenciar persistência de todo o estado do jogo
// - Modelo da rede neural (usando IndexedDB nativo do TensorFlow.js)
// - Replay buffers (reservoir, recent, legacy + totalSeen)
// - Metadata (epsilon, generation, highScore, etc. — expansível)
// ------------------------------------------------------------

import * as tf from '@tensorflow/tfjs';
import experimentConfig from '../../experiment.json';

const EXPERIMENT_NAME = (experimentConfig && experimentConfig.name) ? experimentConfig.name : 'experiment';
const DB_NAME = `FlappyDQNDB_${EXPERIMENT_NAME}`;
const DB_VERSION = 1;
const GAME_DATA_STORE = 'gameData';

export class PersistenceManager {
	constructor() {
		this.modelPath = `indexeddb://flappy-dqn-model-${EXPERIMENT_NAME}`; // Caminho nativo IndexedDB (mais eficiente que localstorage://)
	}

	// ------------------------------------------------------------
	// Abre/conecta ao IndexedDB (cria store se necessário)
	// ------------------------------------------------------------
	async openDB() {
		return new Promise((resolve, reject) => {
			const request = indexedDB.open(DB_NAME, DB_VERSION);

			request.onerror = () => reject(request.error);
			request.onsuccess = () => resolve(request.result);

			request.onupgradeneeded = (e) => {
				const db = e.target.result;
				if (!db.objectStoreNames.contains(GAME_DATA_STORE)) {
					db.createObjectStore(GAME_DATA_STORE);
				}
			};
		});
	}

	// ------------------------------------------------------------
	// Modelo da rede neural
	// ------------------------------------------------------------
	async saveModel(model) {
		try {
			await model.save(this.modelPath);
			//console.log('Modelo DQN salvo no IndexedDB (indexeddb://flappy-dqn-model)');
		} catch (err) {
			console.error('Erro ao salvar modelo:', err);
		}
	}

	async loadModel() {
		try {
			const model = await tf.loadLayersModel(this.modelPath);
			console.log('Modelo DQN carregado do IndexedDB');
			return model;
		} catch (err) {
			console.log('Nenhum modelo encontrado (iniciando novo)');
			return null;
		}
	}

	async deleteModel() {
		try {
			await tf.io.removeModel(this.modelPath);
			console.log('Modelo DQN removido do IndexedDB');
		} catch (err) {
			console.warn('Erro ao remover modelo:', err);
		}
	}

	// ------------------------------------------------------------
	// Dados do jogo (buffers + metadata)
	// ------------------------------------------------------------
	async saveGameData(key, data) {
		const db = await this.openDB();
		const tx = db.transaction(GAME_DATA_STORE, 'readwrite');
		const store = tx.objectStore(GAME_DATA_STORE);
		store.put(data, key);
		await tx.complete;
		//console.log(`Dados salvos no IndexedDB com chave: ${key}`);
	}

	async loadGameData(key) {
		const db = await this.openDB();
		const tx = db.transaction(GAME_DATA_STORE, 'readonly');
		const store = tx.objectStore(GAME_DATA_STORE);
		const request = store.get(key);

		return new Promise((resolve) => {
			request.onsuccess = () => {
				if (request.result) {
					console.log(`Dados carregados do IndexedDB com chave: ${key}`);
				} else {
					console.log(`Nenhum dado encontrado para chave: ${key}`);
				}
				resolve(request.result || null);
			};
			request.onerror = () => resolve(null);
		});
	}

	async deleteGameData(key) {
		const db = await this.openDB();
		const tx = db.transaction(GAME_DATA_STORE, 'readwrite');
		const store = tx.objectStore(GAME_DATA_STORE);
		store.delete(key);
		await tx.complete;
		console.log(`Dados removidos do IndexedDB com chave: ${key}`);
	}

	// ------------------------------------------------------------
	// Métodos de alto nível específicos para o Flappy DQN
	// ------------------------------------------------------------
	async saveReplayBuffers(replayBuffer) {
		const data = {
			reservoirBuffer: replayBuffer.reservoirBuffer,
			recentBuffer: replayBuffer.recentBuffer,
			legacyBuffer: replayBuffer.buffer, // legado
			totalSeen: replayBuffer.totalSeen,
			reservoirSize: replayBuffer.reservoirSize,
			recentSize: replayBuffer.recentSize
		};
		await this.saveGameData('replayBuffers', data);
	}

	async loadReplayBuffers(replayBuffer) {
		const data = await this.loadGameData('replayBuffers');
		if (data) {
			replayBuffer.reservoirBuffer = data.reservoirBuffer || [];
			replayBuffer.recentBuffer = data.recentBuffer || [];
			replayBuffer.buffer = data.legacyBuffer || [];
			replayBuffer.totalSeen = data.totalSeen || 0;
			// Opcional: restaurar tamanhos se mudarem no futuro
			// replayBuffer.reservoirSize = data.reservoirSize || replayBuffer.reservoirSize;
			// replayBuffer.recentSize = data.recentSize || replayBuffer.recentSize;
		}
		return !!data;
	}

	async saveMetadata(metadata) {
		// metadata = { epsilon, generation, highScore?, etc. }
		await this.saveGameData('metadata', metadata);
	}

	async loadMetadata() {
		return await this.loadGameData('metadata');
	}

	// ------------------------------------------------------------
	// Reset completo (útil para botão de reset)
	// ------------------------------------------------------------
	async clearAll() {
		await this.deleteModel();
		await this.deleteGameData('replayBuffers');
		await this.deleteGameData('metadata');
		console.log('Todo o estado do jogo foi resetado (modelo + buffers + metadata)');
	}
}