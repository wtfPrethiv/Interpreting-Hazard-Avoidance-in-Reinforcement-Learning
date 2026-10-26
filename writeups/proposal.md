# Interpreting Hazard Avoidance in Reinforcement Learning

RQ1 asks whether hazard avoidance generalises across superficial colour changes
when an identifying symbol stays reliable. Familiar success may hide reliance on
different cues. Behaviour alone cannot establish an internal mechanism.

Hypotheses: correlated-colour training may produce greater colour sensitivity;
independent-colour training may encourage stable-symbol use. Both policies may
instead generalise, or both may fail to learn. No outcome is assumed.

The current implementation establishes a controlled simulator, PPO policies,
paired behavioural comparisons, and development pilots. We assess success and
safety separately and include safe shortcuts to expose an always-detour strategy.
Later work with the user will investigate representations and causal influence.

This setup tests cue dependence, not abstract understanding of danger. See the
[protocol](experiment_plan.md), [related work](related_work.md), and
[limitations](limitations.md). RQ2 is outside scope.
