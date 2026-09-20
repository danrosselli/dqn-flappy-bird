# RL Flappy Bird

A reinforcement learning agent that learns to play Flappy Bird — entirely in the browser. The game (Phaser 3), the neural networks (TensorFlow.js), the training loops, and the persistence layer (IndexedDB) all run client-side. No server, no pre-trained model shipped.

The project began as a Deep Q-Network (DQN) study and has since grown into a broader reinforcement learning playground, spanning **value-based** methods (DQN) and **policy-based** methods (REINFORCE, Actor-Critic, PPO).

**[Live Demo](https://danrosselli.github.io/rl-flappy-bird/demo/)** · **[Documentation](https://danrosselli.github.io/rl-flappy-bird/)** · **[Experiments](https://danrosselli.github.io/rl-flappy-bird/experiments/)**

---

## How it works

The agent sees 8 normalized values each frame — pipe distances, gap geometry, bird velocity, world speed — and chooses between two actions: idle or flap. The learning algorithm behind that choice varies by experiment:

- **Value-based (DQN)** — the network outputs a Q-value per action. A target network, experience replay (50k-transition buffer), and ε-greedy exploration drive the learning.
- **Policy-based (REINFORCE, Actor-Critic, PPO)** — the network outputs an action probability distribution. The policy is updated directly from episode returns, one-step TD advantages, or a clipped surrogate objective with GAE-lambda.

Weights, replay memory, and exploration state are saved to IndexedDB on every death, so training persists across browser sessions.

```
State (8 values) → Neural Network → Action → Reward → Learn
```

---

## Project structure

```
rl-flappy-bird/
├── experiments/               # Numbered experiment log
│   ├── 001-base/              # DQN baseline, 8-dim state
│   ├── 002-state9/            # 9-dim state (adds birdY)
│   ├── 003-forget-and-grow/   # Forget-and-grow hypothesis
│   ├── 004-td-error-gated-training/
│   ├── 005-filter-batch-with-td-error-threshold/
│   ├── 006-gradient-policy/   # REINFORCE (episodic policy gradient)
│   ├── 007-actor-critic/      # Actor-Critic (online one-step TD)
│   └── 008-ppo/               # PPO (clipped surrogate + GAE-lambda)
├── data/                      # Training history datasets
├── site/                      # Documentation website (Eleventy)
├── vite/                      # Shared Vite configs (dev/prod)
├── public/                    # Shared game assets (sprites, audio)
└── package.json               # Root scripts
```

Each experiment directory is self-contained:

```
experiments/001-base/
├── experiment.json    # Machine-readable config
├── runs/001.json      # Run results
├── README.md          # Hypothesis, conclusion
└── src/               # Experiment source code
    ├── game/          # Phaser 3 game scenes
    ├── rl/            # Agent, replay buffer, persistence
    └── main.js        # Entry point
```

---

## Experiments

| # | Name | Family | State | Best Score | Description |
|---|------|--------|-------|------------|-------------|
| 001 | base | DQN | 8-dim | 485 | Baseline — pipe distances, gap, velocity, speed |
| 002 | state9 | DQN | 9-dim | 456 | Adds absolute bird Y position |
| 003 | forget-and-grow | DQN | 8-dim | — | Forget-and-grow hypothesis (ER decay + network expansion) |
| 004 | td-error-gated-training | DQN | 8-dim | — | Skips updates when the batch TD-error is low |
| 005 | filter-batch-with-td-error-threshold | DQN | 8-dim | — | Filters the batch by a TD-error threshold |
| 006 | gradient-policy | REINFORCE | 8-dim | — | Episodic policy gradient over full trajectories |
| 007 | actor-critic | Actor-Critic | 8-dim | — | Online one-step TD advantage, per-frame updates |
| 008 | ppo | PPO | 8-dim | — | Batched updates, GAE-lambda, clipped surrogate objective |

Each experiment is a self-contained directory with its own `package.json`, source code, config, and results. The documentation site reads `experiment.json` and `runs/*.json` to auto-generate experiment pages.

See [`experiments/README.md`](experiments/README.md) for the full experiment schema and lifecycle.

---

## Getting started

### Prerequisites

- [Bun](https://bun.sh/) (or Node.js 18+)

### Install

```bash
git clone https://github.com/danrosselli/rl-flappy-bird.git
cd rl-flappy-bird
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
cd experiments/008-ppo
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
- **ML**: [TensorFlow.js 4.22](https://www.tensorflow.org/js) — DQN, REINFORCE, Actor-Critic, and PPO agents, WebGL backend
- **Persistence**: IndexedDB — weights, replay memory, epsilon, generation
- **Bundler**: [Vite](https://vitejs.dev/) — dev server + production builds
- **Site**: [Eleventy 3](https://www.11ty.dev/) — static documentation site
- **Runtime**: [Bun](https://bun.sh/) — package manager + scripts

---

## License

[MIT](LICENSE) — Daniel Rosselli
