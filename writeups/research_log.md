# Research log

## 2026-10-26 — implementation and pre-pilot validation

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

## Entry template

- Date:
- Hypothesis or change:
- Configuration / seed / checkpoint / code revision:
- Observation:
- Interpretation:
- Next step:
