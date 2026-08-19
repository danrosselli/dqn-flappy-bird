# DQN Flappy Bird

A Deep Q-Network agent that learns to play Flappy Bird — entirely in the browser. The game (Phaser 3), the neural network (TensorFlow.js), the training loop, and the persistence layer (IndexedDB) all run client-side. No server, no pre-trained model shipped.

**[Live Demo](https://danrosselli.github.io/dqn-flappy-bird/demo/)** · **[Documentation](https://danrosselli.github.io/dqn-flappy-bird/)** · **[Experiments](https://danrosselli.github.io/dqn-flappy-bird/experiments/)**

---

## How it works

The agent sees 8 normalized values each frame — pipe distances, gap geometry, bird velocity, world speed — and outputs two Q-values: idle or flap. A target network, experience replay (50k-transition buffer), and ε-greedy exploration drive the learning. Weights, replay memory, and exploration state are saved to IndexedDB on every death, so training persists across browser sessions.

```
State (8 values) → Neural Network (8→64→64→2) → Q-Idle / Q-Flap → Action → Reward → Replay Memory → Train
```

---

## Project structure

```
dqn-flappy-bird/
├── experiments/               # Numbered experiment log
│   ├── 001-base/              # 8-dimensional state (baseline)
│   │   ├── experiment.json    # Machine-readable config
│   │   ├── runs/001.json      # Run results
│   │   ├── README.md          # Hypothesis, conclusion
│   │   └── src/               # Experiment source code
│   │       ├── game/          # Phaser 3 game scenes
│   │       ├── rl/            # DQN agent, replay buffer, persistence
│   │       └── main.js        # Entry point
│   ├── 002-state9/            # 9-dim state (adds birdY)
│   └── 003-forget-and-grow/  # Forget-and-grow hypothesis
├── data/                      # Training history datasets
├── site/                      # Documentation website (Eleventy)
├── vite/                      # Shared Vite configs (dev/prod)
├── public/                    # Shared game assets (sprites, audio)
└── package.json               # Root scripts
```

---

## Experiments

| # | Name | State | Best Score | Description |
|---|------|-------|------------|-------------|
| 001 | base | 8-dim | 485 | Baseline — pipe distances, gap, velocity, speed |
| 002 | state9 | 9-dim | 456 | Adds absolute bird Y position |
| 003 | forget-and-grow | 8-dim | — | Forget-and-grow hypothesis |

Each experiment is a self-contained directory with its own `package.json`, source code, config, and results. The documentation site reads `experiment.json` and `runs/*.json` to auto-generate experiment pages.

See [`experiments/README.md`](experiments/README.md) for the full experiment schema and lifecycle.

---

## Getting started

### Prerequisites

- [Bun](https://bun.sh/) (or Node.js 18+)

### Install

```bash
git clone https://github.com/danrosselli/dqn-flappy-bird.git
cd dqn-flappy-bird
bun install
```

### Run an experiment

```bash
cd experiments/003-forget-and-grow
bun run dev
```

Opens on `localhost:8080`. The agent starts from scratch — flapping at random, dying immediately — and improves as episodes accumulate. Progress saves to IndexedDB automatically.

Switch to any experiment directory to try a different configuration:

```bash
cd experiments/001-base
bun run dev
```

### Build an experiment

```bash
cd experiments/003-forget-and-grow
bun run build
```

Output goes to `experiments/003-forget-and-grow/dist/`.

### Build all experiments

```bash
bun run build-all
```

### Run the documentation site

```bash
cd site
bun install
bun run dev
```

Opens on `localhost:8081`. The site reads experiment data from `experiments/` and generates pages automatically.

### Build the full site (site + game bundles)

```bash
cd site
bun run build:full
```

This builds all experiments, generates the Eleventy site, and copies game bundles into `_site/game/{slug}/` for the live demo.

---

## Tech stack

- **Game**: [Phaser 3.90](https://phaser.io/) — Arcade physics, Flappy Bird environment
- **ML**: [TensorFlow.js 4.22](https://www.tensorflow.org/js) — DQN agent, WebGL backend
- **Persistence**: IndexedDB — weights, replay memory, epsilon, generation
- **Bundler**: [Vite](https://vitejs.dev/) — dev server + production builds
- **Site**: [Eleventy 3](https://www.11ty.dev/) — static documentation site
- **Runtime**: [Bun](https://bun.sh/) — package manager + scripts

---

## License

[MIT](LICENSE) — Daniel Rosselli
