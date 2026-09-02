/* ============================================================
 * RUNNING MEAN / STD — Welford's online algorithm
 * ------------------------------------------------------------
 * Maintains a running estimate of mean and variance, updated
 * incrementally each training cycle. Used to normalize
 * advantages in a way that is stable across batches of varying
 * size and composition.
 * ============================================================ */

export class RunningMeanStd {
  constructor() {
    this.mean = 0;
    this.var = 1;
    this.count = 0;
  }

  /**
   * Update running statistics with a new batch of values.
   * Uses Welford's online algorithm for numerical stability.
   * @param {number[]} values
   */
  update(values) {
    const n = values.length;
    if (n === 0) return;

    const batchMean = values.reduce((s, v) => s + v, 0) / n;
    const batchVar = values.reduce((s, v) => s + (v - batchMean) ** 2, 0) / n;

    if (this.count === 0) {
      this.mean = batchMean;
      this.var = batchVar;
      this.count = n;
      return;
    }

    const totalCount = this.count + n;
    const delta = batchMean - this.mean;
    const newMean = this.mean + (delta * n) / totalCount;

    const m2A = this.var * this.count;
    const m2B = batchVar * n;
    const m2 = m2A + m2B + (delta * delta * this.count * n) / totalCount;

    this.var = m2 / totalCount;
    this.mean = newMean;
    this.count = totalCount;
  }

  /**
   * Normalize a value using running statistics.
   * @param {number} value
   * @returns {number}
   */
  normalize(value) {
    const std = Math.max(Math.sqrt(this.var), 1e-2);
    return (value - this.mean) / std;
  }
}
