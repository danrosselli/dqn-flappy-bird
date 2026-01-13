/* ------------------------------------------------------------
 * REPLAY BUFFER (com todos os métodos antigos + híbrido novo)
 * ------------------------------------------------------------ */
export class ReplayBuffer {
	constructor(recentSize = 10000, reservoirSize = 40000) {
		this.reservoirSize = reservoirSize;     // Default 50k (memória longa uniforme)
		this.recentSize = recentSize;           // Default 10k (memória curta FIFO)
		this.reservoirBuffer = [];              // Reservoir principal
		this.recentBuffer = [];                 // Buffer recente separado (FIFO)
		this.totalSeen = 0;                     // Contador pro reservoir
		this.buffer = [];                       // Buffer "legado" pra compatibilidade com métodos antigos (FIFO simples)
	}

	add(state, action, reward, nextState, done) {
		const transition = { state, action, reward, nextState, done };

		// Legacy: Mantém o buffer antigo pra métodos como sampleLastPrioritizedAndRandom
		if (this.buffer.length >= (this.reservoirSize + this.recentSize)) {
			this.buffer.shift();
		}
		this.buffer.push(transition);

		// Recent FIFO (sempre as mais novas)
		this.recentBuffer.push(transition);
		if (this.recentBuffer.length > this.recentSize) {
			this.recentBuffer.shift();
		}

		// Reservoir sampling (memória longa uniforme)
		this.totalSeen++;

		if (this.reservoirBuffer.length < this.reservoirSize) {
			this.reservoirBuffer.push(transition);
		} else {
			// Limita totalSeen para manter a chance de substituição em no mínimo 5% (1/20)
			const maxTotalSeen = this.reservoirSize * 20;
			if (this.totalSeen > maxTotalSeen) {
				this.totalSeen = maxTotalSeen;
			}

			const idx = Math.floor(Math.random() * this.totalSeen);
			if (idx < this.reservoirSize) {
				this.reservoirBuffer[idx] = transition;
			}
		}
	}

	// Método antigo 1: Random básico (mantido pra estudo)
	sampleRandomBasic(batchSize) {
		const batch = [];
		for (let i = 0; i < batchSize; i++) {
			const idx = Math.floor(Math.random() * this.buffer.length);
			batch.push(this.buffer[idx]);
		}
		return batch;
	}

	// Método antigo 2: Last + Random (mantido pra estudo — usa o buffer legado)
	sampleLastPrioritizedAndRandom(batchSize) {
		const batch = [];
		const recentPercentage = 0.10;
		const numRecent = Math.floor(batchSize * recentPercentage);
		const numRandom = batchSize - numRecent;

		const recentStart = Math.max(0, this.buffer.length - numRecent);
		for (let i = this.buffer.length - 1; i >= recentStart; i--) {
			batch.push(this.buffer[i]);
		}

		if (numRandom > 0 && this.buffer.length > 0) {
			for (let i = 0; i < numRandom; i++) {
				const idx = Math.floor(Math.random() * this.buffer.length);
				batch.push(this.buffer[idx]);
			}
		}

		// Shuffle final
		for (let i = batch.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[batch[i], batch[j]] = [batch[j], batch[i]];
		}
		return batch;
	}

	// Método antigo 3: Reward prioritized (mantido pra estudo)
	sampleRewardPrioritized(batchSize) {
		if (this.buffer.length === 0) return [];
		const epsilon = 1.0;
		const absRewards = this.buffer.map(t => Math.abs(t.reward) + epsilon);
		const total = absRewards.reduce((a, b) => a + b, 0);
		const probs = absRewards.map(r => r / total);

		const cumsum = []; let sum = 0;
		for (let p of probs) {
			sum += p;
			cumsum.push(sum);
		}

		const batch = [];
		for (let i = 0; i < batchSize; i++) {
			const rand = Math.random();
			let low = 0, high = cumsum.length - 1;
			while (low <= high) {
				const mid = Math.floor((low + high) / 2);
				if (cumsum[mid] >= rand) {
					high = mid - 1;
				} else {
					low = mid + 1;
				}
			}
			const idx = low;
			batch.push(this.buffer[idx]);
		}

		// Shuffle final
		for (let i = batch.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[batch[i], batch[j]] = [batch[j], batch[i]];
		}
		return batch;
	}

	// Novo método principal: Prioridade sequencial nas experiências mais recentes (sempre)
	sampleHybridWithRecentSeq(batchSize) {
		const batch = [];

		// === 1. Sempre adiciona 5% do batch como frames sequenciais mais recentes do recentBuffer ===
		let remainingSize = batchSize;
		const numSeqFrames = Math.floor(batchSize * 0.05); // 5% do batchSize (sempre)

		if (numSeqFrames > 0 && this.recentBuffer.length > 0) {
			// Pega os últimos N frames (mais recentes), em ordem cronológica (mais antigo primeiro)
			const seqStartIdx = Math.max(0, this.recentBuffer.length - numSeqFrames);
			const seqTrajectory = this.recentBuffer.slice(seqStartIdx, this.recentBuffer.length);

			batch.push(...seqTrajectory);
			remainingSize = batchSize - seqTrajectory.length; // Ajusta caso recentBuffer seja curto
		}

		// === 2. Preenche o resto mantendo a proporção original recent/reservoir ===
		const recentRatio = 0.35; // Mesma proporção anterior (35% recent shuffled)
		const numRecent = Math.floor(remainingSize * recentRatio);
		const numReservoir = remainingSize - numRecent;

		// Recent shuffled (amostras aleatórias do recentBuffer inteiro)
		if (numRecent > 0 && this.recentBuffer.length > 0) {
			const recentCopy = [...this.recentBuffer];
			// Shuffle simples
			for (let i = recentCopy.length - 1; i > 0; i--) {
				const j = Math.floor(Math.random() * (i + 1));
				[recentCopy[i], recentCopy[j]] = [recentCopy[j], recentCopy[i]];
			}
			batch.push(...recentCopy.slice(0, numRecent));
		}

		// Reservoir (uniforme de longo prazo)
		if (numReservoir > 0 && this.reservoirBuffer.length > 0) {
			for (let i = 0; i < numReservoir; i++) {
				const idx = Math.floor(Math.random() * this.reservoirBuffer.length);
				batch.push(this.reservoirBuffer[idx]);
			}
		}

		// === 3. Shuffle final do batch inteiro (exceto a parte sequencial que já está em ordem) ===
		for (let i = batch.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[batch[i], batch[j]] = [batch[j], batch[i]];
		}

		// Caso raro: preenche com reservoir extra se ainda faltar
		while (batch.length < batchSize && this.reservoirBuffer.length > 0) {
			const idx = Math.floor(Math.random() * this.reservoirBuffer.length);
			batch.push(this.reservoirBuffer[idx]);
		}

		return batch;
	}

	// Novo método principal: Foco exclusivo no último crash (trajetória terminal mais recente)
	sampleHybridWithCrashFocus(batchSize) {
		const batch = [];

		// === 1. Busca último crash nas últimas 10 entradas do recentBuffer ===
		let lastTerminalIdx = -1;
		const lookback = 10;
		const startIdx = Math.max(0, this.recentBuffer.length - lookback);
		for (let i = this.recentBuffer.length - 1; i >= startIdx; i--) {
			if (this.recentBuffer[i].done) {
				lastTerminalIdx = i;
				break;
			}
		}

		// === 2. Se houver crash recente: adiciona 20% do batch da trajetória do crash ===
		let remainingSize = batchSize;
		if (lastTerminalIdx !== -1) {
			const numCrashFrames = Math.floor(batchSize * 0.10); // 10% do batchSize
			const crashStartIdx = Math.max(0, lastTerminalIdx - numCrashFrames + 1);
			const crashTrajectory = this.recentBuffer.slice(crashStartIdx, lastTerminalIdx + 1);

			batch.push(...crashTrajectory);
			remainingSize = batchSize - crashTrajectory.length; // pode ser >80% se trajetória curta
		}

		// === 3. Preenche o resto mantendo a proporção original recent/reservoir ===
		const recentRatio = 0.35; // mesma proporção do fallback anterior
		const numRecent = Math.floor(remainingSize * recentRatio);
		const numReservoir = remainingSize - numRecent;

		// Recent (shuffled)
		if (numRecent > 0 && this.recentBuffer.length > 0) {
			const recentCopy = [...this.recentBuffer];
			// Shuffle simples
			for (let i = recentCopy.length - 1; i > 0; i--) {
				const j = Math.floor(Math.random() * (i + 1));
				[recentCopy[i], recentCopy[j]] = [recentCopy[j], recentCopy[i]];
			}
			batch.push(...recentCopy.slice(0, numRecent));
		}

		// Reservoir (uniforme)
		if (numReservoir > 0 && this.reservoirBuffer.length > 0) {
			for (let i = 0; i < numReservoir; i++) {
				const idx = Math.floor(Math.random() * this.reservoirBuffer.length);
				batch.push(this.reservoirBuffer[idx]);
			}
		}

		// === 4. Shuffle final do batch inteiro ===
		for (let i = batch.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[batch[i], batch[j]] = [batch[j], batch[i]];
		}

		// Caso raro de batch menor que batchSize (trajetória muito curta + buffers vazios),
		// preenchemos com reservoir extra se disponível
		while (batch.length < batchSize && this.reservoirBuffer.length > 0) {
			const idx = Math.floor(Math.random() * this.reservoirBuffer.length);
			batch.push(this.reservoirBuffer[idx]);
		}

		return batch;
	}

	sampleHybrid(batchSize) {
		const batch = [];

		// Proporção fixa: 30–40% das amostras recentes, resto do reservoir (memória longa)
		const recentRatio = 0.35;
		const numRecent = Math.floor(batchSize * recentRatio);
		const numReservoir = batchSize - numRecent;

		// Amostras recentes (shuffle para não ter viés temporal forte)
		if (numRecent > 0 && this.recentBuffer.length > 0) {
			const recentCopy = [...this.recentBuffer];
			// Shuffle simples
			for (let i = recentCopy.length - 1; i > 0; i--) {
				const j = Math.floor(Math.random() * (i + 1));
				[recentCopy[i], recentCopy[j]] = [recentCopy[j], recentCopy[i]];
			}
			batch.push(...recentCopy.slice(0, numRecent));
		}

		// Amostras do reservoir (uniforme longa prazo)
		if (numReservoir > 0 && this.reservoirBuffer.length > 0) {
			for (let i = 0; i < numReservoir; i++) {
				const idx = Math.floor(Math.random() * this.reservoirBuffer.length);
				batch.push(this.reservoirBuffer[idx]);
			}
		}

		// Shuffle final do batch inteiro
		for (let i = batch.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[batch[i], batch[j]] = [batch[j], batch[i]];
		}

		return batch;
	}

	// Recentes — pega as ÚLTIMAS numRecent (mais novas primeiro, sem shuffle)
	/*
	if (numRecent > 0 && this.recentBuffer.length > 0) {
			const recentStart = Math.max(0, this.recentBuffer.length - numRecent);
			for (let i = this.recentBuffer.length - 1; i >= recentStart; i--) {
					batch.push(this.recentBuffer[i]);  // Ordem cronológica reversa (mais nova primeiro)
			}
	}
	*/

	size() {
		return this.reservoirBuffer.length + this.recentBuffer.length;
	}
}