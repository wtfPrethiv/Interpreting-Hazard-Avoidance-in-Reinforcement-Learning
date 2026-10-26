# Related work and design implications

This is an exploratory synthetic-environment study. The literature below informs
its design; it does not establish novelty or predict a successful mechanism.

| Primary source | Relevant result or method | Implication and boundary here |
| --- | --- | --- |
| [Geirhos et al. (2020), Shortcut Learning](https://arxiv.org/abs/2004.07780) | Standard-distribution performance can conceal brittle decision rules. | Compare paired colour shifts with stable symbols. A null difference is valid. |
| [Langosco et al. (2022), Goal Misgeneralization](https://proceedings.mlr.press/v162/langosco22a.html) | RL policies can retain capabilities while pursuing an unintended goal under distribution shift. | Measure success and safety separately. Cue-dependent errors alone do not prove goal misgeneralisation. |
| [Mini et al. (2023), Understanding and Controlling a Maze-Solving Policy Network](https://arxiv.org/abs/2310.08043) | Activation interventions partially control a maze policy; representations can be distributed and redundant. | Direct RL interpretability precedent. Do not assume dedicated hazard neurons or claim that this project introduces policy patching. |
| [Trim and Grayston (2024), Mechanistic Interpretability of RL Agents](https://arxiv.org/abs/2411.00867) | Examines features and navigation biases in a procedural-maze policy. | Another related RL analysis; our hazard/status/colour controls require separate empirical validation. |
| [Hewitt and Liang (2019), Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/) | Probe performance needs controls for the probe's own capacity. | Later probes need simple models, held-out families, and control labels. Recoverability is not causal use. |
| [Zhang and Nanda (2024), Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) | Patching results depend on metrics and corruption methods. | Later interventions need matched donors, fixed metrics, and damage controls. Evidence is from language models; transfer here must be tested. |
| [Heimersheim and Nanda (2024), How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255) | Discusses interpretation and metric pitfalls. | Whole-vector changes cannot automatically be called hazard-specific. |
| [Henderson et al. (2018), Deep RL that Matters](https://arxiv.org/abs/1709.06560) | Reproducibility and experimental variance matter for RL comparisons. | Log exact configurations and evaluate independent training seeds. |
| [Agarwal et al. (2021), Statistical Precipice](https://arxiv.org/abs/2108.13264) | Few-run point estimates conceal considerable uncertainty. | Pilot intervals describe layouts within one seed; they do not estimate between-policy variability. |

The proposed contribution is a controlled investigation connecting cue shifts,
behaviour, and eventually internal interventions in a small hazard-avoidance
policy. Its distinctiveness remains provisional. An exhaustive novelty review,
and interpretation of learned mechanisms, are not completed by this reading list.

Implementation references: [Gymnasium environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/),
[SB3 image conventions](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html),
[SB3 custom policies and encoder sharing](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html),
[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
