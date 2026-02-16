// persistenceManager.js
// ------------------------------------------------------------
// Classe única para gerenciar persistência de todo o estado do jogo
// - Modelo da rede neural (NestedModel custom ou TF.js padrão)
// - Replay buffers (reservoir, recent, legacy + totalSeen)
// - Metadata (epsilon, generation, highScore, etc. — expansível)
// ------------------------------------------------------------

import * as tf from '@tensorflow/tfjs';

const DB_NAME = 'FlappyDQNDB';
const DB_VERSION = 2; // Incrementado para suportar NestedModel
const GAME_DATA_STORE = 'gameData';

export class PersistenceManager {
	constructor() {
		this.modelPath = 'indexeddb://flappy-dqn-model'; // Para modelos TF.js padrão (legado)
		this.nestedModelPath = 'nested-model-v1'; // Chave para NestedModel serializado
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
	// NESTED MODEL (Novo formato)
	// ------------------------------------------------------------

	/**
	 * Salva NestedModel serializado
	 */
	async saveNestedModel(modelData) {
		try {
			await this.saveGameData(this.nestedModelPath, modelData);
			console.log('NestedModel salvo no IndexedDB');
		} catch (err) {
			console.error('Erro ao salvar NestedModel:', err);
		}
	}

	/**
	 * Carrega NestedModel serializado
	 */
	async loadNestedModel() {
		try {
			const data = await this.loadGameData(this.nestedModelPath);
			if (data) {
				console.log('NestedModel carregado do IndexedDB');
			}
			return data;
		} catch (err) {
			console.log('Nenhum NestedModel encontrado');
			return null;
		}
	}

	/**
	 * Remove NestedModel salvo
	 */
	async deleteNestedModel() {
		await this.deleteGameData(this.nestedModelPath);
		console.log('NestedModel removido do IndexedDB');
	}

	// ------------------------------------------------------------
	// Modelo TF.js padrão (Legado - mantido para compatibilidade)
	// ------------------------------------------------------------
	async saveModel(model) {
		try {
			await model.save(this.modelPath);
		} catch (err) {
			console.error('Erro ao salvar modelo TF.js:', err);
		}
	}

	async loadModel() {
		try {
			const model = await tf.loadLayersModel(this.modelPath);
			console.log('Modelo TF.js legado carregado');
			return model;
		} catch (err) {
			return null;
		}
	}

	async deleteModel() {
		try {
			await tf.io.removeModel(this.modelPath);
		} catch (err) {
			console.warn('Erro ao remover modelo TF.js:', err);
		}
	}

	// ------------------------------------------------------------
	// Dados do jogo (buffers + metadata)
	// ------------------------------------------------------------
	async saveGameData(key, data) {
		const db = await this.openDB();
		return new Promise((resolve, reject) => {
			const tx = db.transaction(GAME_DATA_STORE, 'readwrite');
			const store = tx.objectStore(GAME_DATA_STORE);
			const request = store.put(data, key);

			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}

	async loadGameData(key) {
		const db = await this.openDB();
		return new Promise((resolve) => {
			const tx = db.transaction(GAME_DATA_STORE, 'readonly');
			const store = tx.objectStore(GAME_DATA_STORE);
			const request = store.get(key);

			request.onsuccess = () => resolve(request.result || null);
			request.onerror = () => resolve(null);
		});
	}

	async deleteGameData(key) {
		const db = await this.openDB();
		return new Promise((resolve, reject) => {
			const tx = db.transaction(GAME_DATA_STORE, 'readwrite');
			const store = tx.objectStore(GAME_DATA_STORE);
			const request = store.delete(key);

			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}

	// ------------------------------------------------------------
	// Métodos de alto nível específicos para o Flappy DQN
	// ------------------------------------------------------------
	async saveReplayBuffers(replayBuffer) {
		// Verifica se é StratifiedPER (tem buffers múltiplos) ou PER padrão
		const isStratified = replayBuffer.constructor.name === 'StratifiedPER';

		const data = {
			type: isStratified ? 'stratified' : 'standard',
			reservoirBuffer: replayBuffer.reservoirBuffer || [],
			recentBuffer: replayBuffer.recentBuffer || [],
			legacyBuffer: replayBuffer.buffer || [],
			totalSeen: replayBuffer.totalSeen || 0,
			reservoirSize: replayBuffer.reservoirSize || 40000,
			recentSize: replayBuffer.recentSize || 10000,
			// StratifiedPER specific
			strataStats: isStratified ? replayBuffer.getStats?.() : null
		};

		await this.saveGameData('replayBuffers', data);
	}

	async loadReplayBuffers(replayBuffer) {
		const data = await this.loadGameData('replayBuffers');
		if (!data) return false;

		// Restaura dados comuns
		replayBuffer.reservoirBuffer = data.reservoirBuffer || [];
		replayBuffer.recentBuffer = data.recentBuffer || [];
		replayBuffer.buffer = data.legacyBuffer || [];
		replayBuffer.totalSeen = data.totalSeen || 0;

		if (data.reservoirSize) replayBuffer.reservoirSize = data.reservoirSize;
		if (data.recentSize) replayBuffer.recentSize = data.recentSize;

		return true;
	}

	async saveMetadata(metadata) {
		await this.saveGameData('metadata', metadata);
	}

	async loadMetadata() {
		return await this.loadGameData('metadata');
	}

	// ------------------------------------------------------------
	// Reset completo (útil para botão de reset)
	// ------------------------------------------------------------
	async clearAll() {
		// Limpa NestedModel
		await this.deleteNestedModel();

		// Limpa modelo TF.js legado (se existir)
		await this.deleteModel();

		// Limpa dados auxiliares
		await this.deleteGameData('replayBuffers');
		await this.deleteGameData('metadata');

		console.log('Todo o estado do jogo foi resetado');
	}
}