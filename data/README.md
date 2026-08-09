# Data

This directory contains machine-generated and manually collected data produced by the DQN Flappy Bird project.

The purpose of this directory is to preserve information that is useful for:

* analyzing training;
* comparing experiments;
* generating charts;
* measuring learning progress;
* reproducing results;
* providing data to the project website.

The data should generally be stored in machine-readable formats such as JSON or CSV.

---

# Directory Structure

The initial structure is:

```text
data/
├── training/
├── scores/
├── metrics/
└── README.md
```

Additional directories can be created as the project evolves.

---

# Training Data

The `training/` directory contains information generated during the training process.

Possible structure:

```text
data/
└── training/
    ├── experiment-001.json
    ├── experiment-002.json
    └── experiment-003.json
```

A training dataset can contain information for every episode.

Example:

```json
{
  "experiment": 3,
  "episodes": [
    {
      "episode": 1,
      "score": 0,
      "reward": -10,
      "steps": 37,
      "epsilon": 1.0,
      "loss": 0.82
    },
    {
      "episode": 2,
      "score": 1,
      "reward": 1,
      "steps": 84,
      "epsilon": 0.995,
      "loss": 0.74
    }
  ]
}
```

For long training sessions, this can become very large. The format and storage strategy may therefore evolve.

---

# Scores

The `scores/` directory contains information specifically related to game performance.

Example:

```text
data/
└── scores/
    ├── experiment-001.json
    ├── experiment-002.json
    └── experiment-003.json
```

Example data:

```json
{
  "experiment": 3,
  "best": 217,
  "average": 143,
  "median": 139,
  "runs": 1000
}
```

Scores can be used to generate comparisons such as:

```text
Experiment    Best Score    Average Score
------------------------------------------
001           87            42
002           143           71
003           217           143
004           189           121
```

---

# Metrics

The `metrics/` directory contains additional measurements that help analyze the learning process.

Possible metrics include:

* reward;
* loss;
* epsilon;
* Q-values;
* survival time;
* number of steps;
* exploration rate;
* exploitation rate;
* episode duration;
* training time.

Example:

```json
{
  "experiment": 7,
  "metrics": {
    "averageReward": 18.42,
    "averageLoss": 0.034,
    "averageQValue": 2.71,
    "averageSurvivalTime": 42.7,
    "finalEpsilon": 0.05
  }
}
```

---

# Experiment Identification

Every dataset associated with an experiment should contain an experiment identifier.

Example:

```json
{
  "experiment": 7
}
```

This allows data to be associated with:

```text
experiments/007/
```

and with the Git tag:

```text
experiment-007
```

The experiment number is the primary link between documentation and data.

---

# Runs

An experiment may contain multiple training runs.

For example, Experiment 007 might be executed five times:

```text
Experiment 007

Run 1 → Best score: 812
Run 2 → Best score: 847
Run 3 → Best score: 791
Run 4 → Best score: 901
Run 5 → Best score: 833
```

In this situation, data can be organized as:

```text
data/
└── training/
    └── experiment-007/
        ├── run-001.json
        ├── run-002.json
        ├── run-003.json
        ├── run-004.json
        └── run-005.json
```

A run should identify both the experiment and itself:

```json
{
  "experiment": 7,
  "run": 4,
  "episodes": 100000,
  "bestScore": 901
}
```

This distinction becomes important when statistical comparisons are required.

---

# Raw Data vs Processed Data

When possible, distinguish between raw data and processed data.

For example:

```text
data/
├── raw/
│   └── training/
│
├── processed/
│   ├── scores/
│   └── metrics/
│
└── README.md
```

Raw data represents what the training system actually produced.

Processed data represents information derived from the raw data for analysis or visualization.

This makes it possible to regenerate charts and statistics if the processing method changes.

---

# Large Data Files

Not every generated file should necessarily be committed to Git.

Training can produce very large datasets.

For example:

```text
10,000 episodes
100,000 episodes
1,000,000 episodes
```

can generate significant amounts of data.

Small datasets that are useful for understanding or reproducing an experiment can be committed normally.

Large datasets may eventually require an external storage strategy.

Possible future solutions include:

* Git LFS;
* release assets;
* external object storage;
* compressed datasets;
* storing only representative samples.

The decision should be made based on the actual size of the project's datasets.

---

# Data and the Website

The website can consume data from this directory to generate visualizations.

For example:

```text
data/
└── scores/
    └── experiment-007.json
```

could generate a chart:

```text
Score
900 |                         *
800 |                    *    *
700 |                *   *    *
600 |            *   *   *    *
500 |        *   *   *   *    *
400 |    *   *   *   *   *    *
300 | *  *   *   *   *   *    *
    +----------------------------
       Training Episodes
```

The same data can be used to compare multiple experiments.

---

# Data Principles

Data stored in this directory should preferably be:

1. machine-readable;
2. clearly associated with an experiment;
3. reproducible;
4. versioned when reasonably sized;
5. documented;
6. independent of presentation whenever possible.

The data layer should describe the experiment rather than contain website-specific formatting.
