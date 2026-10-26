# Research log

## 2026-10-26 â€” implementation and pre-pilot validation

- Preserved the original scaffold in commit `2ac8545` before implementation.
- Restricted this phase to RQ1 behaviour. Interpretability modules stay deferred
  for joint design after pilot review; RQ2 is excluded.
- Added safe-shortcut cases to expose an always-detour strategy. This is a
  deliberate development design choice, not an empirical finding.
- Adopted Python 3.12, SB3 PPO, a small shared CNN, and an image-only observation.
- Verified all 256 scenario families and both hazard statuses against route-value
  calculations; checked split symmetry isolation and seeded scenario reproduction.
- All 20 tests passed, including environment contracts, appearance-only trajectory
  invariance, hazard re-entry, model save/reload, and baseline checks. Lint and
  formatting checks passed. A 1024-transition smoke policy failed readiness as
  expected; that run validates plumbing and provides no learning evidence.
- Visually inspected paired environment rendering and smoke figures. Corrected
  figure/evaluation evidence labels so smoke results remain distinguishable.
- Main protocol remains unfrozen. Next: benchmark, run one bounded pilot per
  condition, and record the actual competence results without interpreting mechanisms.

## 2026-10-26 — bounded pilot outcomes

- Suite: `pilot_suite_20261026T135243493782`; seed 0 per condition; 999,424 transitions each.
- Total suite time: 925.625 seconds, within the 7,200-second budget.
- Both complete paired evaluations failed readiness. Familiar hazardous-case entry
  was 38.3% (correlated training) and 54.7% (randomised training); overall goal
  success was 88.5% and 88.3%. Safe-shortcut success was 96.9% for both.
- All five reference baselines completed. Symbol and oracle references reached
  100% success with no hazard entry in every evaluated cue condition.
- Retained all checkpoints and trajectories. Visually inspected the final figure;
  it distinguishes hazardous-case success from safe-shortcut competence.
- No settings changed in response to these outcomes. No main runs, discovery-data
  analysis, probe fitting, or neural interventions were performed.
- Manifests truthfully flag the working tree dirty: generated results were
  untracked; at first launch the only source difference was two formatting blank
  lines, committed as `a33e5bd`. Both runs used functionally identical source.
- Proposed next action: review failure traces and consider a separately versioned,
  longer development run before changing other factors. This was not executed.
- Full numerical results and limitations: [RQ1 pilot findings](rq1_findings.md).

## Entry template

- Date:
- Hypothesis or change:
- Configuration / seed / checkpoint / code revision:
- Observation:
- Interpretation:
- Next step:
