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
    this.nextIdx = 0;
    this.size_ = 0;
    this.maxPriority = 1.0;
    this.priorityEpsilon = 1e-5;
  }

  add(state, action, reward, nextState, done) {
    const transition = { state, action, reward, nextState, done };
    if (state[0] >= 1) return;

    const idx = this.nextIdx;
    transition.index = idx;  // ← Adiciona index permanente na transition

    if (this.size_ < this.bufferSize) {
      this.buffer.push(transition);
      this.size_++;
    } else {
      this.buffer[idx] = transition;
    }
    this.nextIdx = (this.nextIdx + 1) % this.bufferSize;

    const priority = this.maxPriority + this.priorityEpsilon;
    const powered = Math.pow(priority, this.alpha);
    this.tree.update(idx, powered);
  }

  sample(batchSize, beta) {
    const batch = [];
    const total = this.tree.total();
    if (total === 0) return batch;  // ← Retorna só batch vazio

    const segment = total / batchSize;
    for (let i = 0; i < batchSize; i++) {
      const a = segment * i;
      const b = segment * (i + 1);
      let s = Math.random() * (b - a) + a;
      const idx = this.tree.get(s);
      if (idx < 0 || idx >= this.size_) continue;

      const transition = this.buffer[idx];
      const prob = this.tree.getLeaf(idx) / total;
      const weight = Math.pow(this.size_ * prob, -beta);

      // ← Adiciona propriedades na transition (mutável, pois objeto)
      transition.weight = weight;  // Para importance sampling no train()

      batch.push(transition);
    }

    // Normaliza weights pelo max (estabilidade)
    if (batch.length > 0) {
      const maxWeight = Math.max(...batch.map(t => t.weight));
      batch.forEach(t => t.weight = t.weight / maxWeight);
    }

    // Shuffle (mantém como antes)
    for (let i = batch.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [batch[i], batch[j]] = [batch[j], batch[i]];
    }

    return batch;  // ← Agora retorna SÓ o batch (com .index e .weight dentro)
  }

  updatePriorities(batch) {  // ← Agora recebe o batch modificado
    for (const t of batch) {
      if (t.tdError === undefined) continue;  // Segurança
      const tdErrorAbs = Math.abs(t.tdError);
      const priority = tdErrorAbs + this.priorityEpsilon;
      this.maxPriority = Math.max(this.maxPriority, tdErrorAbs);
      const powered = Math.pow(priority, this.alpha);
      this.tree.update(t.index, powered);  // Usa o index salvo na transition
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