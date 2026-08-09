# Project Website

This directory contains the source code for the public website of the DQN Flappy Bird project.

The website documents the project, explains the reinforcement learning implementation, presents experiments and results, and publishes insights obtained during the development of the agent.

The website is generated using **Eleventy** and deployed automatically using **GitHub Actions** to **GitHub Pages**.

---

# Purpose

The website is not the game itself.

The repository contains two related but distinct components:

```text
DQN Flappy Bird
│
├── Game and AI
│   └── src/
│
└── Documentation and Project Website
    └── site/
```

The game contains the implementation.

The website explains the implementation and documents its evolution.

---

# Directory Structure

The planned structure is:

```text
site/
├── src/
│   ├── _data/
│   ├── _includes/
│   │   ├── layouts/
│   │   └── components/
│   │
│   ├── index.md
│   │
│   ├── about/
│   │   └── index.md
│   │
│   ├── how-it-works/
│   │   ├── index.md
│   │   ├── game.md
│   │   ├── agent.md
│   │   ├── state.md
│   │   ├── actions.md
│   │   ├── rewards.md
│   │   ├── dqn.md
│   │   └── training.md
│   │
│   ├── experiments/
│   │   └── index.md
│   │
│   └── blog/
│       └── index.md
│
├── assets/
│   ├── css/
│   ├── js/
│   └── images/
│
├── eleventy.config.js
├── package.json
└── README.md
```

This structure is a starting point and may evolve as the website grows.

---

# Source Content

The `site/src/` directory contains the source content used by Eleventy.

Markdown should be preferred for documentation and articles.

Example:

```text
site/src/blog/
├── index.md
├── first-training.md
├── reward-function.md
└── improving-exploration.md
```

A post can contain front matter:

```markdown
---
title: "Improving the Reward Function"
description: "An analysis of how reward design affected training."
date: 2026-08-09
tags:
  - reinforcement-learning
  - experiments
  - rewards
---

# Improving the Reward Function

The reward function has a significant impact on how the agent learns.
```

Eleventy converts this Markdown into HTML.

---

# Layouts

The `_includes/` directory contains reusable templates.

Example:

```text
site/src/_includes/
├── layouts/
│   ├── base.njk
│   ├── page.njk
│   ├── post.njk
│   └── experiment.njk
│
└── components/
    ├── header.njk
    ├── footer.njk
    ├── navigation.njk
    ├── experiment-card.njk
    └── score-chart.njk
```

Layouts allow pages to share common elements without duplicating HTML.

---

# Site Data

The `_data/` directory contains structured data used to generate pages.

Example:

```text
site/src/_data/
├── site.json
├── navigation.json
└── project.json
```

Example `site.json`:

```json
{
  "title": "DQN Flappy Bird",
  "description": "Teaching an agent to play Flappy Bird using Deep Q-Learning.",
  "author": "Daniel Rosselli"
}
```

Example `project.json`:

```json
{
  "algorithm": "DQN",
  "game": "Flappy Bird",
  "status": "active",
  "repository": "danrosselli/dqn-flappy-bird"
}
```

---

# Experiments on the Website

The website should eventually be able to consume experiment documentation directly from:

```text
experiments/
```

For example:

```text
experiments/007/
├── README.md
├── config.json
└── results.json
```

Eleventy can use these files to generate a public page such as:

```text
/experiments/007/
```

containing:

* experiment objective;
* hypothesis;
* configuration;
* code changes;
* training information;
* results;
* charts;
* conclusions;
* Git reference.

The goal is to avoid duplicating experiment information specifically for the website.

---

# Data and Visualization

The website may consume data from:

```text
data/
```

to generate visualizations.

Potential visualizations include:

* score progression;
* average score;
* reward progression;
* loss;
* epsilon decay;
* survival time;
* comparison between experiments;
* training performance.

Chart.js may be used for interactive charts.

The website should preferably consume raw or structured data and generate the visualization dynamically rather than storing screenshots of charts.

---

# Build Process

The website source is not published directly.

Eleventy generates a static version of the site.

The build process is:

```text
Markdown
   +
Nunjucks templates
   +
JSON data
   +
CSS
   +
JavaScript
        │
        ▼
     Eleventy
        │
        ▼
     _site/
        │
        ▼
GitHub Pages
```

The `_site/` directory contains generated files and should normally not be committed to Git.

It should be included in `.gitignore`:

```text
site/_site/
```

---

# GitHub Pages

The website will be hosted as a GitHub Pages project site.

The expected URL is:

```text
https://danrosselli.github.io/dqn-flappy-bird/
```

The repository itself remains:

```text
https://github.com/danrosselli/dqn-flappy-bird
```

GitHub Actions will build the site and deploy the generated Eleventy output to GitHub Pages.

The repository does not need to contain the generated HTML files.

---

# GitHub Actions

The expected deployment flow is:

```text
git push
   │
   ▼
GitHub Actions
   │
   ├── Checkout repository
   ├── Install dependencies
   ├── Build Eleventy site
   ├── Generate site/_site
   ├── Upload Pages artifact
   └── Deploy to GitHub Pages
```

This means the website is automatically updated whenever changes are pushed to the configured branch.

---

# Path Prefix

Because this is a GitHub Pages project site rather than a user site, the website is served under:

```text
/dqn-flappy-bird/
```

The Eleventy configuration must therefore account for this path prefix when generating links to:

* CSS;
* JavaScript;
* images;
* internal pages;
* fonts;
* other assets.

The configuration will use Eleventy's `pathPrefix` functionality.

---

# Relationship Between Repository Components

The overall repository architecture is:

```text
dqn-flappy-bird/
│
├── src/
│   └── Game and AI implementation
│
├── experiments/
│   └── Experiment documentation and metadata
│
├── data/
│   └── Training and analysis data
│
├── site/
│   └── Public website source
│
└── .github/
    └── Automated workflows
```

Each component has a specific responsibility.

```text
src
 ↓
Implementation

experiments
 ↓
Scientific documentation

data
 ↓
Measurements and results

site
 ↓
Public presentation

Git
 ↓
Complete source-code history
```

---

# Design Principle

The website should document the evolution of the project rather than simply describe its final state.

The project is an ongoing experiment in reinforcement learning and continuous learning.

The website should therefore make it possible to understand:

```text
What was tried?
       ↓
Why was it tried?
       ↓
What changed?
       ↓
What happened?
       ↓
What was learned?
       ↓
What was tried next?
```

The history of the project is part of the project itself.
