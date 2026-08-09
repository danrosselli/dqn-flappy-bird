# Experiments

This directory contains the documentation, configuration, metadata, and results associated with each machine learning experiment.

An experiment represents a specific hypothesis or change to the DQN Flappy Bird agent. An experiment can involve anything from a small parameter adjustment to a major architectural change involving multiple source files.

The Git repository is responsible for preserving the exact state of the source code used by an experiment. This directory is responsible for documenting **why the experiment was performed, how it was configured, what happened, and what was learned from it**.

---

## Experiment Structure

Each experiment should have a sequential numeric identifier:

```text
experiments/
├── 001/
│   ├── README.md
│   ├── config.json
│   └── results.json
│
├── 002/
│   ├── README.md
│   ├── config.json
│   └── results.json
│
└── 003/
    ├── README.md
    ├── config.json
    └── results.json
```

The experiment number should never be reused.

Once an experiment has been created, its number becomes a permanent identifier.

---

# Experiment README

The `README.md` file is the human-readable documentation of the experiment.

It should explain:

* what was being investigated;
* the hypothesis;
* what changed;
* why the change was made;
* how the experiment was executed;
* the relevant results;
* observations;
* conclusions;
* possible future experiments.

Example:

```markdown
# Experiment 007

## Objective

Evaluate whether introducing a target network improves training stability.

## Hypothesis

A target network should reduce oscillations in the Q-value estimates
and produce more stable learning.

## Changes

- Added TargetNetwork class.
- Added periodic target network updates.
- Modified DQN training logic.
- Added target update interval configuration.

## Training

Episodes: 100000
Learning rate: 0.001
Gamma: 0.95

## Results

Best score: 847
Average score: 421

## Conclusion

The target network produced more stable training and improved the
best score compared with Experiment 006.

## Git Reference

Tag: experiment-007
```

---

# Configuration

The `config.json` file contains the parameters used for the experiment.

The purpose is to make the experiment configuration machine-readable.

Example:

```json
{
  "experiment": 7,

  "agent": {
    "algorithm": "DQN",
    "learningRate": 0.001,
    "gamma": 0.95,
    "epsilon": {
      "initial": 1.0,
      "minimum": 0.05,
      "decay": 0.995
    }
  },

  "network": {
    "inputSize": 6,
    "hiddenLayers": [
      64,
      64
    ],
    "outputSize": 2
  },

  "training": {
    "episodes": 100000,
    "batchSize": 64,
    "replayMemorySize": 50000,
    "targetUpdateInterval": 1000
  },

  "rewards": {
    "passPipe": 1,
    "death": -10,
    "survival": 0
  }
}
```

The configuration can evolve as the project evolves. Not every experiment needs to contain exactly the same fields.

For example, an experiment that introduces a new neural-network architecture may add:

```json
{
  "network": {
    "architecture": "dueling-dqn",
    "hiddenLayers": [
      128,
      128,
      64
    ]
  }
}
```

The configuration should describe the parameters that are relevant to reproducing or understanding the experiment.

---

# Results

The `results.json` file contains machine-readable results produced by the experiment.

Example:

```json
{
  "experiment": 7,

  "training": {
    "episodes": 100000,
    "durationSeconds": 8421
  },

  "score": {
    "best": 847,
    "average": 421,
    "median": 398
  },

  "performance": {
    "bestEpisode": 87321,
    "averageSurvivalTime": 42.7,
    "maximumSurvivalTime": 91.4
  },

  "learning": {
    "finalEpsilon": 0.05,
    "averageLoss": 0.034
  }
}
```

Additional metrics can be added when they become useful.

---

# Training History

For experiments that generate a large amount of training information, results can reference a separate dataset.

For example:

```text
experiments/
└── 007/
    ├── README.md
    ├── config.json
    ├── results.json
    └── training.json
```

A training history could contain:

```json
{
  "experiment": 7,
  "episodes": [
    {
      "episode": 1,
      "score": 0,
      "reward": -10,
      "loss": 0.91,
      "epsilon": 1.0
    },
    {
      "episode": 2,
      "score": 1,
      "reward": 1,
      "loss": 0.87,
      "epsilon": 0.995
    }
  ]
}
```

For very large datasets, this information should normally be stored in `data/` rather than directly inside the experiment directory.

---

# Git and Experiments

Git is responsible for preserving the source-code state of an experiment.

An experiment may contain major structural changes:

* new files;
* removed files;
* renamed files;
* new classes;
* new algorithms;
* modified neural networks;
* changes to the game environment;
* changes to the training process.

These changes do not need to be copied into `experiments/`.

Instead, Git preserves the exact source-code state.

An experiment should normally have a corresponding Git tag:

```text
experiment-001
experiment-002
experiment-003
```

For example:

```text
Experiment #007
        │
        ├── README.md
        ├── config.json
        ├── results.json
        │
        └── Git tag
              │
              ▼
        experiment-007
              │
              ▼
        Exact source code
```

This creates a direct relationship between the documentation and the implementation.

---

# Experiment Lifecycle

A typical experiment follows this process:

```text
main
  │
  ▼
Create experiment branch
  │
  ▼
Implement changes
  │
  ▼
Train agent
  │
  ▼
Analyze results
  │
  ▼
Document experiment
  │
  ▼
Commit final state
  │
  ▼
Create Git tag
  │
  ▼
experiment-XXX
```

If the experiment is successful, its changes may be merged into `main`.

If it is unsuccessful, the branch does not need to be merged. The tag and documentation still preserve the experiment.

---

# Reproducibility

An experiment should contain enough information to answer:

1. What was changed?
2. Why was it changed?
3. What parameters were used?
4. How much training was performed?
5. What was the result?
6. Which version of the source code produced the result?
7. What was learned?

The combination of:

```text
README.md
config.json
results.json
Git tag
```

should provide this information.

---

# Relationship With the Website

The website can use the contents of this directory to automatically generate experiment pages.

For example:

```text
experiments/007/
├── README.md
├── config.json
└── results.json
```

can become:

```text
Website
└── Experiments
    └── Experiment #007
        ├── Objective
        ├── Configuration
        ├── Results
        ├── Charts
        └── Conclusions
```

This means that documenting an experiment in the repository can automatically become documentation on the public website.
