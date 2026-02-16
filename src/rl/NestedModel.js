import * as tf from '@tensorflow/tfjs';

export class NestedModel {
	constructor() {
		this.step = 0;

		// Arquitetura: 8 → 64 → 64 → 64 → 32 → 2
		this.layers = [];

		// Camada 0: 8→64 (LENTA - memória longa)
		this.layers.push(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [8] }));

		// Camada 1: 64→64 (MÉDIA)  
		this.layers.push(tf.layers.dense({ units: 64, activation: 'relu' }));

		// Camada 2: 64→64 (RÁPIDA)
		this.layers.push(tf.layers.dense({ units: 64, activation: 'relu' }));

		// Camada 3: 64→32 (NORMAL)
		this.layers.push(tf.layers.dense({ units: 32, activation: 'relu' }));

		// Output: 32→2 (NORMAL)
		this.layers.push(tf.layers.dense({ units: 2, activation: 'linear' }));

		// Otimizadores escalonados (Rampa)
		this.optimizers = [
			{ opt: tf.train.adam(0.00001), freq: 100, name: 'Lento' },    // Layer 0
			{ opt: tf.train.adam(0.0001), freq: 10, name: 'Médio' },    // Layer 1  
			{ opt: tf.train.adam(0.001), freq: 1, name: 'Rápido' },   // Layer 2
			{ opt: tf.train.adam(0.001), freq: 1, name: 'Normal' },   // Layer 3
			{ opt: tf.train.adam(0.001), freq: 1, name: 'Normal' }    // Output
		];

		console.log(`NestedModel criado: ${this.layers.length} camadas, ${this.optimizers.length} otimizadores`);
	}

	forward(state) {
		let x = state;
		for (const layer of this.layers) {
			x = layer.apply(x);
		}
		return x;
	}

	predict(state) {
		return tf.tidy(() => this.forward(state));
	}

	getQValues(state) {
		return tf.tidy(() => {
			const qValues = this.forward(state);
			return { 0: qValues.dataSync()[0], 1: qValues.dataSync()[1] };
		});
	}

	async trainStep(states, targetQ, actionsTensor, sampleWeights) {
		this.step++;

		if (this.optimizers.length !== this.layers.length) {
			console.error(`Mismatch: ${this.layers.length} layers vs ${this.optimizers.length} optimizers`);
			return;
		}

		// Clona tensores de entrada para garantir que não serão modificados/disposed
		// Isso é crucial porque o minimize() pode manter referências
		const statesClone = states.clone();
		const targetQClone = targetQ.clone();
		const actionsClone = actionsTensor.clone();
		const weightsClone = sampleWeights.clone();

		for (let i = 0; i < this.layers.length; i++) {
			const optimizerConfig = this.optimizers[i];

			if (!optimizerConfig || !optimizerConfig.opt) continue;

			const { opt, freq, name } = optimizerConfig;

			if (this.step % freq === 0) {
				// FIX: Use trainableWeights (Variables) instead of getWeights (Tensors/Snapshots)
				// LayerVariable wraps the actual tf.Variable in .val
				const vars = this.layers[i].trainableWeights.map(w => w.val);
				if (vars.length === 0) continue;

				try {
					// Loss function SEM tf.tidy() - minimize() já gerencia os gradientes
					const lossFn = () => {
						const qValues = this.forward(statesClone);
						const oneHotActions = tf.oneHot(actionsClone, 2);
						const selectedQ = tf.sum(qValues.mul(oneHotActions), 1);
						const diff = targetQClone.sub(selectedQ);
						return tf.mean(diff.square().mul(weightsClone));
					};

					opt.minimize(lossFn, false, vars);

				} catch (e) {
					console.error(`Erro no otimizador ${i} (${name}):`, e);
					throw e;
				}

				if (this.step % 500 === 0) {
					console.log(`Step ${this.step}: Camada ${i} (${name}) atualizada`);
				}
			}
		}

		// Limpa os clones após todas as atualizações
		statesClone.dispose();
		targetQClone.dispose();
		actionsClone.dispose();
		weightsClone.dispose();
	}

	getWeights() {
		const weights = [];
		for (const layer of this.layers) {
			weights.push(...layer.getWeights());
		}
		return weights;
	}

	setWeights(newWeights) {
		let weightIndex = 0;
		for (const layer of this.layers) {
			const layerWeightCount = layer.getWeights().length;
			const layerNewWeights = newWeights.slice(weightIndex, weightIndex + layerWeightCount);

			// Clona para evitar compartilhamento de memória
			const clonedWeights = layerNewWeights.map(t => t.clone());
			layer.setWeights(clonedWeights);

			weightIndex += layerWeightCount;
		}
	}

	syncFrom(sourceModel) {
		const sourceWeights = sourceModel.getWeights();
		// We do not dispose sourceWeights because experience shows it can break the source model
		// if getWeights returns view tensors or if backend handles memory differently.
		// Safe approach: just set.
		this.setWeights(sourceWeights);
	}

	syncTargetModel(targetModel) {
		targetModel.syncFrom(this);
	}

	/**
	 * Serialização segura que evita disposed tensors
	 */
	serialize() {
		const weightsData = [];

		for (let i = 0; i < this.layers.length; i++) {
			const layer = this.layers[i];
			const layerWeights = layer.getWeights();
			const layerData = [];

			for (let j = 0; j < layerWeights.length; j++) {
				const w = layerWeights[j];

				// Verificação explícita de disposed
				if (!w || w.isDisposed) {
					console.warn(`Tensor disposed na camada ${i}, peso ${j}. Usando zeros.`);
					// Cria array de zeros do tamanho esperado
					const size = layer.units * (layer.kernelSize || 1); // Fallback
					layerData.push(new Float32Array(size).fill(0));
					continue;
				}

				try {
					// Clona, extrai dados, descarta clone
					const data = tf.tidy(() => {
						const clone = w.clone();
						return clone.dataSync();
					});
					layerData.push(Array.from(data));
				} catch (e) {
					console.error(`Erro ao serializar peso [${i}][${j}]:`, e);
					layerData.push([]);
				}
			}

			weightsData.push(layerData);
		}

		return {
			architecture: 'nested-8-64-64-64-32-2',
			step: this.step,
			weights: weightsData,
			optimizers: this.optimizers.map(o => ({
				freq: o.freq,
				name: o.name
			}))
		};
	}

	async deserialize(data) {
		if (!data || !data.weights) {
			console.warn('Dados inválidos para desserialização');
			return false;
		}

		try {
			for (let i = 0; i < this.layers.length && i < data.weights.length; i++) {
				const layerData = data.weights[i];
				if (!layerData || layerData.some(arr => !arr || arr.length === 0)) {
					console.warn(`Camada ${i} com dados inválidos, pulando`);
					continue;
				}

				const tensors = layerData.map(arr => tf.tensor(arr));
				this.layers[i].setWeights(tensors);
			}

			this.step = data.step || 0;
			console.log(`NestedModel carregado: ${data.weights.length} camadas, step ${this.step}`);
			return true;
		} catch (e) {
			console.error('Erro ao desserializar NestedModel:', e);
			return false;
		}
	}

	getInfo() {
		const totalParams = this.layers.reduce((sum, layer) => {
			return sum + layer.getWeights().reduce((s, w) => s + w.size, 0);
		}, 0);

		return {
			step: this.step,
			totalParams,
			layers: this.layers.map((layer, i) => ({
				index: i,
				units: layer.units,
				activation: layer.activation?.name || 'linear',
				optimizer: this.optimizers[i].name,
				freq: this.optimizers[i].freq,
				params: layer.getWeights().reduce((s, w) => s + w.size, 0)
			}))
		};
	}
}