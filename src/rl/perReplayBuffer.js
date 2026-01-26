/* ------------------------------------------------------------
 * PER REPLAY BUFFER (Prioritized Experience Replay com buffer simples)
 * ------------------------------------------------------------
 * Versão simplificada baseada no ReplayBuffer original, mas sem híbrido (apenas um buffer circular de 50k).
 * Mantém descarte de estados iniciais e add circular como no legacy buffer.
 * Sample retorna {batch, indices, weights} para uso no train() do DQNAgent.
 * Indices são números simples (0 a bufferSize-1).
 * ------------------------------------------------------------ */
export class PERReplayBuffer {
  constructor(alpha = 0.6) {
    this.bufferSize = 50000;
    this.alpha = alpha;

    this.buffer = [];

    this.tree = new SumTree(this.bufferSize);

    this.nextIdx = 0; // Ponteiro circular
    this.size_ = 0; // Tamanho atual

    this.maxPriority = 1.0; // Prioridade inicial máxima

    this.priorityEpsilon = 1e-5;

  }

  add(state, action, reward, nextState, done) {
    const transition = { state, action, reward, nextState, done };

    // Descarte estados iniciais onde ainda não há canos (dx >=1.0)
    if (state[0] >= 1) {
      return;
    }

    // Adiciona ao buffer (circular)
    const idx = this.nextIdx;
    if (this.size_ < this.bufferSize) {
      this.buffer.push(transition);
      this.size_++;
    } else {
      this.buffer[idx] = transition;
    }
    this.nextIdx = (this.nextIdx + 1) % this.bufferSize;

    // Atualiza prioridade inicial na tree
    const priority = this.maxPriority + this.priorityEpsilon; // ou um valor fixo alto como 1.0
    const powered = Math.pow(priority, this.alpha);
    this.tree.update(idx, powered);
  }

  sample(batchSize, beta) {
    const batch = [];
    const indices = []; // Números simples (0 a bufferSize-1)
    const weights = []; // Para Importance Sampling

    const total = this.tree.total();
    if (total === 0) return { batch, indices, weights }; // Buffer vazio

    const segment = total / batchSize;
    for (let i = 0; i < batchSize; i++) {
      const a = segment * i;
      const b = segment * (i + 1);
      let s = Math.random() * (b - a) + a;
      const idx = this.tree.get(s);
      if (idx < 0 || idx >= this.size_) continue; // Raro erro de índice

      const transition = this.buffer[idx];
      batch.push(transition);
      indices.push(idx);

      // Peso IS: w = (N * P(i)) ^ (-beta)
      const prob = this.tree.getLeaf(idx) / total;
      let weight = Math.pow(this.size_ * prob, -beta);
      weights.push(weight);
    }

    // Normaliza weights pelo máximo (estabilidade)
    if (weights.length > 0) {
      const maxWeight = Math.max(...weights);
      weights.forEach((w, i) => weights[i] = w / maxWeight);
    }

    // Shuffle batch (opcional, mas mantém como em alguns métodos originais)
    for (let i = batch.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [batch[i], batch[j]] = [batch[j], batch[i]];
      // Troca também indices e weights para manter alinhamento
      [indices[i], indices[j]] = [indices[j], indices[i]];
      [weights[i], weights[j]] = [weights[j], weights[i]];
    }

    return { batch, indices, weights };
  }

  updatePriorities(indices, tdErrors) {
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const tdErrorAbs = Math.abs(tdErrors[i]);
      const priority = tdErrorAbs + this.priorityEpsilon;
      this.maxPriority = Math.max(this.maxPriority, tdErrorAbs); // mantém o max sem o epsilon
      const powered = Math.pow(priority, this.alpha);
      this.tree.update(idx, powered);
    }
  }

  size() {
    return this.size_;
  }
}

// Classe SumTree (mesma do exemplo anterior)
class SumTree {
  constructor(capacity) {
    this.capacity = capacity;
    this.tree = new Array(2 * capacity).fill(0);
  }

  update(idx, priority) {
    idx += this.capacity; // Folha
    this.tree[idx] = priority;
    while (idx > 1) {
      idx = Math.floor(idx / 2);
      this.tree[idx] = this.tree[2 * idx] + this.tree[2 * idx + 1];
    }
  }

  total() {
    return this.tree[1] || 0;
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

  getLeaf(idx) {
    return this.tree[idx + this.capacity];
  }
}