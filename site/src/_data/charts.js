/* ============================================================
 * Derived chart data computed from the real training constants
 * in src/rl/dqn.js (epsilon decay schedule).
 * ============================================================ */

const EPSILON_START = 0.9;
const EPSILON_DECAY = 0.9995;
const EPSILON_MIN = 0;

function epsilonAt(step) {
  return Math.max(EPSILON_START * Math.pow(EPSILON_DECAY, step), EPSILON_MIN);
}

export default function () {
  const maxStep = 16000;
  const samples = 100;
  const epsilonCurve = [];
  for (let i = 0; i <= samples; i++) {
    const step = Math.round((maxStep / samples) * i);
    epsilonCurve.push({ step, epsilon: epsilonAt(step) });
  }

  /* Useful reference markers for annotations */
  const stepToReach = (target) =>
    Math.ceil(Math.log(target / EPSILON_START) / Math.log(EPSILON_DECAY));

  return {
    epsilonCurve,
    epsilonMarkers: [
      { label: "ε = 0.50", step: stepToReach(0.5), epsilon: 0.5 },
      { label: "ε = 0.10", step: stepToReach(0.1), epsilon: 0.1 },
      { label: "ε = 0.02", step: stepToReach(0.02), epsilon: 0.02 },
    ],
    epsilonMaxStep: maxStep,
  };
}
