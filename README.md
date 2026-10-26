# Interpreting Hazard Avoidance in Reinforcement Learning

RQ1: **Do agents use colour shortcuts or a reliable hazard symbol, and does their
avoidance generalise when colours change?** This exploratory study separates task
success from safety. RQ2 and interpretability experiments are outside the current
implementation. Named policy activation points are available for later work together.

## Setup

Use Python 3.12 from the repository root. On this Windows machine:

```powershell
py -V:Astral/CPython3.12.15 -m venv .venv
# Optional CUDA build, used for the recorded local pilots:
.venv/Scripts/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e '.[dev]'
.venv/Scripts/python -m pytest -q
```

On other systems use `python3.12 -m venv .venv`, activate it, then run
`python -m pip install -e '.[dev]'`. CPU operation is supported. The optional
`notebooks` extra installs Jupyter. `requirements-tested.txt` records the exact
local dependency versions; reproduce CUDA wheels with the explicit PyTorch index
above. Cross-device bitwise reproducibility is not promised.

Editable installation exposes the generic packages inside `src`, for example
`from envs.gridworld import HazardGridworld`. There is no project-specific package
directory, no manual `sys.path` modification, and no need to set `PYTHONPATH`.

## Reproducible commands

Commands below assume the virtual environment is active; alternatively replace
`python` with `.venv/Scripts/python` on Windows.

```powershell
# Short end-to-end smoke check. Use a fresh run ID when repeating.
python scripts/train.py --config configs/smoke.yaml --run-id smoke_demo --kind smoke
python scripts/evaluate.py --checkpoint results/runs/smoke_demo/final.zip --output results/metrics/smoke_demo.json

# CPU/GPU benchmark, then one pilot per condition. Includes evaluation and baselines.
python scripts/run_pilots.py --budget-seconds 7200

# Standalone training and paired evaluation.
python scripts/train.py --config configs/rq1_correlated.yaml --seed 0 --max-seconds 3000
python scripts/train.py --config configs/rq1_randomised.yaml --seed 0 --max-seconds 3000
python scripts/evaluate.py --checkpoint results/runs/RUN_ID/final.zip --config configs/rq1_appearance.yaml
python scripts/evaluate.py --baseline symbol --config configs/smoke.yaml
python scripts/make_figures.py results/metrics/EVALUATION_A.json results/metrics/EVALUATION_B.json
```

Configuration precedence is base YAML, repeated `--config` overlays, then repeated
`--set dotted.key=value` overrides. Checkpoint evaluation starts from its saved
training configuration. Unknown keys fail rather than silently taking effect.
Output directories and evaluation files refuse overwrites.

## Environment and measurements

The agent sees a channel-first `uint8` RGB image of an 11x11 grid, with 8-pixel
tiles. Actions are up, down, left, right. A short route and a longer detour lead
to the goal. Half of training episodes have a hazardous shortcut; half have a
safe shortcut, preventing an always-detour policy from demonstrating cue use.

X marks a hazard and a circle marks a safe comparison tile. Colours are sampled
once per episode: correlated training uses the conventional red-hazard/blue-safe
assignment with probability 0.95 per marked tile; randomised training samples
each colour independently of status. Reversed evaluation swaps the conventional
mapping completely. The symbol remains reliable in all primary conditions.

Goal entry gives +1 task reward, every step costs 0.02, and each hazard entry
costs 1 safety unit. Learning reward is task reward minus safety cost. Leaving
and re-entering a hazard costs again; a blocked move while on it does not.
Hazards permit continuation. Goal entry terminates; 100 steps truncate. There
is no bonus. Safety costs, task returns, learning returns, and their discounted
counterparts are recorded separately. The oracle verifies the intended incentives.

The scenario manifest reserves 128 training, 32 development, 48 discovery, and
48 final-evaluation layout families. Rotations, reflections, and cue/status pairs
stay within a family. Pilots use development data only. See
[the protocol](writeups/experiment_plan.md) for evaluation and readiness criteria.

PPO uses a small CNN shared by actor and critic, then separate heads. Labels in
simulator `info` and the scenario manifest are never fed into the neural policy.
Reference colour/symbol/path/oracle policies have privileged scenario access and
are explicitly baselines, not learned agents.

## Outputs and current boundaries

- `results/manifests/`: configuration, scenario split, seed, revision, dependency,
  device, checkpoint hashes, and pilot-suite records.
- `results/metrics/`: episode records, replayable action sequences, decision-state
  probabilities, paired differences, and family-clustered uncertainty.
- `results/figures/`: static figures generated from recorded evaluations.
- `results/runs/`: ignored checkpoints and per-environment training CSVs. Keep
  these locally or back them up separately; hashes alone cannot recover a model.
- `writeups/`: protocol, related work, research log, findings, and limitations.
- `notebooks/`: result inspection only; implementation lives in source and scripts.

Behavioural pilots do not establish internal representations or causal mechanisms.
Probe and intervention modules remain deferred placeholders. Main experiments
require a protocol review after competent pilots; they are never launched by the
pilot runner. Multiple training seeds are needed before group-level conclusions.
