/* ------------------------------------------------------------
 * STRATIFIED PER (Prioritized Experience Replay Híbrido)
 * ------------------------------------------------------------
 * Divide o buffer em 3 estratos por dificuldade/score:
 * - Estrato 0 (Fácil): Score 0-100 ou cano longe (50% do batch)
 * - Estrato 1 (Médio): Score 100-250 ou cano médio (30% do batch)  
 * - Estrato 2 (Difícil): Score 250+ ou cano perto (20% do batch)
 * 
 * Isso evita que o catastrophic forgetting domine o treinamento.
 * ------------------------------------------------------------ */

import * as tf from '@tensorflow/tfjs';

// SumTree (mesma implementação que você já usa)
class SumTree {
	constructor(capacity) {
		this.capacity = capacity;
		this.tree = new Float32Array(2 * capacity);
		this.data = new Array(capacity);
		this.writeIdx = 0;
		this.size_ = 0;
	}

	update(idx, priority) {
		idx += this.capacity;
		this.tree[idx] = priority;
		while (idx > 1) {
			idx = Math.floor(idx / 2);
			this.tree[idx] = this.tree[2 * idx] + this.tree[2 * idx + 1];
		}
	}

	add(priority, data) {
		const idx = this.writeIdx;
		this.data[idx] = data;
		this.update(idx, priority);
		this.writeIdx = (this.writeIdx + 1) % this.capacity;
		if (this.size_ < this.capacity) this.size_++;
		return idx;
	}

	get(s) {
		let idx = 1;
		while (idx < this.capacity) {
			const left = 2 * idx;
			if (s <= this.tree[left]) {
				idx = left;
			} else {
				s -= this.tree[left];
				idx = left + 1;
			}
		}
		return idx - this.capacity;
	}

	total() {
		return this.tree[1] || 0;
	}

	getPriority(idx) {
		return this.tree[idx + this.capacity];
	}
}

export class StratifiedPER {
	constructor(capacityPerStrata = 15000, alpha = 0.6) {
		this.alpha = alpha;
		this.epsilon = 1e-5;

		// 3 estratos com suas próprias árvores e dados
		this.stratas = [
			{
				name: 'easy',   // Score 0-100
				tree: new SumTree(capacityPerStrata),
				maxPriority: 1.0,
				size: 0
			},
			{
				name: 'medium', // Score 100-250
				tree: new SumTree(capacityPerStrata),
				maxPriority: 1.0,
				size: 0
			},
			{
				name: 'hard',   // Score 250+
				tree: new SumTree(capacityPerStrata),
				maxPriority: 1.0,
				size: 0
			}
		];

		this.capacity = capacityPerStrata;
	}

	/**
	 * Adiciona transição ao buffer apropriado
	 * @param {number} score - Score atual do jogo (IMPORTANTE: passar this.score)
	 */
	add(state, action, reward, nextState, done, score = null) {
		// Determina estrato baseado no score ou heurística de distância
		let strataIdx = 0;

		if (score !== null) {
			// Método preciso: pelo score acumulado
			if (score < 100) strataIdx = 0;
			else if (score < 250) strataIdx = 1;
			else strataIdx = 2;
		} else {
			// Fallback: heurística pela distância X (state[0] = dx/1058)
			// Quanto menor dx, mais perto do cano = mais difícil
			const dx = state[0];
			if (dx > 0.5) strataIdx = 0;      // Longe (fácil)
			else if (dx > 0.25) strataIdx = 1; // Médio
			else strataIdx = 2;                // Perto (difícil)
		}

		const strata = this.stratas[strataIdx];

		// Cria objeto de transição com metadados para atualização posterior
		const transition = {
			state, action, reward, nextState, done,
			_strataIdx: strataIdx,  // Marca de qual estrato veio
			_absError: strata.maxPriority // Guarda para referência
		};

		// Adiciona à árvore com prioridade máxima atual (+ epsilon para garantir)
		const priority = strata.maxPriority + this.epsilon;
		const idx = strata.tree.add(Math.pow(priority, this.alpha), transition);

		// Guarda índice local para updates (necessário para o updatePriorities)
		transition._treeIdx = idx;

		strata.size = strata.tree.size_;
	}

	/**
	 * Amostra batch mantendo proporção fixa entre estratos
	 * 40% fácil, 40% médio, 20% difícil (evita forgetting)
	 */
	sample(batchSize, beta = 0.4) {
		// Distribuição fixa: garante representação mesmo se TD error for baixo

		const easy = Math.floor(batchSize * 0.5);
		const medium = Math.floor(batchSize * 0.3);
		const hard = batchSize - easy - medium;

		const counts = [easy, medium, hard];

		const batch = [];
		let totalSamples = 0;

		// Amostra de cada estrato independentemente
		this.stratas.forEach((strata, sIdx) => {
			const nSamples = counts[sIdx];
			if (nSamples === 0 || strata.size === 0) return;

			const total = strata.tree.total();
			if (total === 0) return;

			const segment = total / nSamples;

			for (let i = 0; i < nSamples; i++) {
				const a = segment * i;
				const b = segment * (i + 1);
				const s = Math.random() * (b - a) + a;

				const idx = strata.tree.get(s);
				const transition = strata.tree.data[idx];

				if (!transition) continue;

				// Calcula importance sampling weight
				const prob = strata.tree.getPriority(idx) / total;
				const weight = Math.pow(strata.size * prob, -beta);

				transition.weight = weight;
				transition._strataIdx = sIdx; // Garante que sabemos de qual veio
				transition._treeIdx = idx;    // Índice para update

				batch.push(transition);
				totalSamples++;
			}
		});

		// Normaliza weights pelo máximo (estabilidade numérica)
		if (batch.length > 0) {
			const maxWeight = Math.max(...batch.map(t => t.weight));
			batch.forEach(t => t.weight = t.weight / maxWeight);
		}

		// Shuffle final para misturar os estratos (evita bias de ordem)
		for (let i = batch.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[batch[i], batch[j]] = [batch[j], batch[i]];
		}

		return batch;
	}

	/**
	 * Atualiza prioridades baseado no TD Error calculado no treino
	 */
	updatePriorities(batch) {
		for (const transition of batch) {
			if (transition.tdError === undefined) continue;
			if (transition._strataIdx === undefined) continue;

			const tdErrorAbs = Math.abs(transition.tdError);
			const strata = this.stratas[transition._strataIdx];

			// Atualiza prioridade máxima do estrato
			strata.maxPriority = Math.max(strata.maxPriority, tdErrorAbs);

			// Atualiza árvore
			const priority = tdErrorAbs + this.epsilon;
			strata.tree.update(transition._treeIdx, Math.pow(priority, this.alpha));
		}
	}

	size() {
		return this.stratas.reduce((sum, s) => sum + s.size, 0);
	}

	// Método auxiliar para debug (ver distribuição)
	getStats() {
		return this.stratas.map((s, i) => ({
			estrato: ['Fácil(0-100)', 'Médio(100-250)', 'Difícil(250+)'][i],
			size: s.size,
			maxPriority: s.maxPriority.toFixed(4)
		}));
	}
}