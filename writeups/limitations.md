# Limitations

- Synthetic, fully observed, deterministic movement in two-route corridors does
  not imply generalisation to robots, language models, or real-world safety.
- All layout splits share the same rectangular-corridor generator. Split
  isolation and orientation balance reduce leakage; they do not establish
  out-of-generator generalisation or eliminate every geometric heuristic.
- Hazard status is perfectly associated with its reliable symbol. Symbol
  recognition cannot establish abstract knowledge of harmful consequences.
- Colour, symbol size, contrast, and convolutional inductive biases may affect
  learning. A null result is not evidence that shortcut learning is impossible.
- A one-seed pilot can test feasibility, not establish a training-condition
  population difference. Layout-bootstrap intervals omit training-seed variation.
- Safe-shortcut controls expose always-detour behaviour, but their small path
  efficiency advantage may still be difficult for PPO to learn within the budget.
- Reference baselines use privileged state/geometry access. They are checks of
  the intended task, not architecture-matched learned comparisons.
- Deterministic route choices and decision-state action probabilities describe
  different quantities. Blocked actions and mixed-route episodes are recorded.
- Local CUDA/CPU seeds support repeatability, not universal bitwise identity.
  Checkpoints are ignored by Git and require a separate backup to survive deletion.
- No representations have been analysed and no neural interventions implemented.
  Later probe accuracy alone will not establish causal use. Interventions will
  require controls for confounds and general navigation damage.
