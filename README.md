# Recognising Danger

Research scaffold for interpreting hazard avoidance in reinforcement learning.

## Research question

**RQ1:** Does an agent represent hazards in a way that generalises across changes in their visual appearance?

This is a working question to refine before experiments. Reward tradeoffs and a second research question are outside the current scope.

## Structure

- `configs/`: shared settings and RQ1 appearance interventions.
- `src/danger_study/`: environments, agents, evaluation, and interpretability modules.
- `scripts/`: experiment entry-point placeholders.
- `notebooks/`: environment checks, baselines, RQ1 analysis, and final figures.
- `writeups/`: proposal, experiment plan, research log, findings, and limitations.
- `tests/`: environment and intervention test placeholders.
- `results/`: run manifests, metrics, and figures.

## Setup and reproduction

Requires Python 3.10 or newer. From the project directory:

```sh
python -m venv .venv
python -m pip install -e ".[dev,notebooks]"
```

Activate the virtual environment before installing. This repository currently contains a scaffold; training, evaluation, and analysis are not implemented. Dependencies will be added as implementation choices are made.

The intended workflow is to train a baseline, evaluate held-out hazard appearances, collect activations, then run probes and causal interventions. For every run, record the resolved configuration, random seed, code revision, and checkpoint identifier under `results/manifests/`. Define held-out appearances before training and report results across multiple seeds.
