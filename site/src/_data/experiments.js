/* ============================================================
 * EXPERIMENTS DATA LAYER
 * ------------------------------------------------------------
 * Reads experiment metadata directly from the repository:
 *
 *   experiments/
 *   └── 001/
 *       ├── README.md      → objective / hypothesis / conclusion
 *       ├── config.json    → machine-readable configuration
 *       ├── results.json   → machine-readable results
 *       └── training.json  → optional per-episode series
 *
 * Adding experiments/004/ is enough for a new experiment page
 * to be generated automatically on the next build.
 *
 * While the repository contains no experiments, the site renders
 * clearly-marked PLACEHOLDER entries so the full interface can
 * be previewed. Placeholder data is never presented as real.
 * ============================================================ */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
const EXPERIMENTS_DIR = path.join(REPO_ROOT, "experiments");

/* ------------------------------------------------------------
 * Minimal README parser: extracts the title and the first
 * paragraph of known sections (Objective, Hypothesis, Conclusion).
 * ---------------------------------------------------------- */
function parseReadme(markdown) {
  const out = { title: null, objective: null, hypothesis: null, conclusion: null };
  if (!markdown) return out;

  const titleMatch = markdown.match(/^#\s+(.+)$/m);
  if (titleMatch) out.title = titleMatch[1].trim();

  const sections = markdown.split(/^##\s+/m).slice(1);
  const firstParagraph = (body) => {
    const block = body
      .split(/\n\s*\n/)
      .map((s) => s.trim())
      .find((s) => s && !s.startsWith("#") && !s.startsWith("-") && !s.startsWith("```"));
    return block ? block.replace(/\n/g, " ") : null;
  };

  for (const section of sections) {
    const nl = section.indexOf("\n");
    const name = (nl === -1 ? section : section.slice(0, nl)).trim().toLowerCase();
    const body = nl === -1 ? "" : section.slice(nl + 1);
    if (name.startsWith("objective")) out.objective = firstParagraph(body);
    if (name.startsWith("hypothesis")) out.hypothesis = firstParagraph(body);
    if (name.startsWith("conclusion")) out.conclusion = firstParagraph(body);
  }
  return out;
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

/* Deterministic pseudo-series so placeholder charts render without
   pretending to be real training data. */
function demoSeries(seed, episodes, from, to) {
  const points = [];
  let x = seed;
  const rand = () => {
    x = (x * 16807) % 2147483647;
    return (x - 1) / 2147483646;
  };
  const steps = 60;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const base = from + (to - from) * Math.pow(t, 0.6);
    const noise = (rand() - 0.5) * (to - from) * 0.35 * (1 - t * 0.6);
    points.push({
      episode: Math.round(t * episodes),
      score: Math.max(0, Math.round(base + noise)),
    });
  }
  return points;
}

const PLACEHOLDER_EXPERIMENTS = [
  {
    id: "001",
    slug: "001",
    title: "Baseline DQN",
    placeholder: true,
    objective: "Establish a baseline training run with the initial reward function and network.",
    hypothesis: null,
    conclusion: null,
    changes: [],
    config: null,
    results: { score: { best: 87, average: 42 }, training: { episodes: 10000 }, learning: { finalEpsilon: 0.31 } },
    series: demoSeries(11, 10000, 0, 87),
    gitTag: null,
  },
  {
    id: "002",
    slug: "002",
    title: "Gap proximity shaping",
    placeholder: true,
    objective: "Test whether a Gaussian proximity reward around the gap center accelerates early learning.",
    hypothesis: null,
    conclusion: null,
    changes: [],
    config: null,
    results: { score: { best: 143, average: 71 }, training: { episodes: 20000 }, learning: { finalEpsilon: 0.18 } },
    series: demoSeries(23, 20000, 2, 143),
    gitTag: null,
  },
  {
    id: "003",
    slug: "003",
    title: "Two-pipe lookahead state",
    placeholder: true,
    objective: "Extend the state vector with the next pipe so the agent can plan two obstacles ahead.",
    hypothesis: null,
    conclusion: null,
    changes: [],
    config: null,
    results: { score: { best: 217, average: 143 }, training: { episodes: 50000 }, learning: { finalEpsilon: 0.06 } },
    series: demoSeries(37, 50000, 5, 217),
    gitTag: null,
  },
  {
    id: "004",
    slug: "004",
    title: "Speed-aware state",
    placeholder: true,
    objective: "Add normalized pipe speed to the state so the policy can adapt as the world accelerates.",
    hypothesis: null,
    conclusion: null,
    changes: [],
    config: null,
    results: { score: { best: 327, average: 184 }, training: { episodes: 50000 }, learning: { finalEpsilon: 0.05 } },
    series: demoSeries(53, 50000, 8, 327),
    gitTag: null,
  },
];

function loadRealExperiments() {
  if (!fs.existsSync(EXPERIMENTS_DIR)) return [];

  const dirs = fs
    .readdirSync(EXPERIMENTS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^\d+$/.test(d.name))
    .map((d) => d.name)
    .sort();

  const experiments = [];

  for (const dir of dirs) {
    const expDir = path.join(EXPERIMENTS_DIR, dir);
    const readmePath = path.join(expDir, "README.md");
    const readme = fs.existsSync(readmePath)
      ? fs.readFileSync(readmePath, "utf8")
      : null;

    const parsed = parseReadme(readme);
    const config = readJson(path.join(expDir, "config.json"));
    const results = readJson(path.join(expDir, "results.json"));
    const training = readJson(path.join(expDir, "training.json"));

    experiments.push({
      id: dir,
      slug: dir,
      title: parsed.title || `Experiment ${dir}`,
      placeholder: false,
      objective: parsed.objective,
      hypothesis: parsed.hypothesis,
      conclusion: parsed.conclusion,
      changes: [],
      config,
      results,
      series: Array.isArray(training?.episodes) ? training.episodes : null,
      gitTag: `experiment-${dir}`,
      readmeAvailable: !!readme,
    });
  }

  return experiments;
}

export default function () {
  const real = loadRealExperiments();
  const usingDemo = real.length === 0;
  const list = usingDemo ? PLACEHOLDER_EXPERIMENTS : real;

  const maxBest = Math.max(
    ...list.map((e) => e.results?.score?.best ?? 0),
    1
  );

  return {
    usingDemo,
    list,
    count: list.length,
    latest: list[list.length - 1] ?? null,
    recent: list.slice(-2).reverse(),
    maxBest,
  };
}
