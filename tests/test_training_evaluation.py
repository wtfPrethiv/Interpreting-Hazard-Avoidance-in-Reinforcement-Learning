import json

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents.policy import action_probabilities, activation_points
from agents.train import train
from envs.gridworld import HazardGridworld
from evaluation.metrics import paired_differences, readiness, summarise
from evaluation.rollouts import evaluate
from experiments.config import ROOT, load_config


def test_sb3_environment_contract():
    check_env(HazardGridworld())


def test_policy_training_save_reload_and_provenance(tmp_path):
    config = load_config(
        [ROOT / "configs/smoke.yaml"],
        [f"results_dir={tmp_path.as_posix()}", "training.total_steps=128"],
    )
    manifest = train(config, kind="smoke", run_id="test")
    assert manifest["status"] == "completed"
    assert manifest["actual_steps"] == 128
    assert manifest["encoder_shared_actor_critic"]
    model = PPO.load(manifest["checkpoint"], device="cpu")
    obs, _ = HazardGridworld().reset(seed=3)
    before = action_probabilities(model, obs)
    model.save(tmp_path / "roundtrip.zip")
    after = action_probabilities(PPO.load(tmp_path / "roundtrip.zip"), obs)
    np.testing.assert_allclose(before, after, atol=1e-7)
    assert set(activation_points(model.policy)) == {
        "encoder.conv1",
        "encoder.conv2",
        "encoder.conv3",
        "encoder.features",
        "actor.hidden",
    }
    stored = json.loads((tmp_path / "manifests/test.json").read_text())
    assert len(stored["checkpoint_sha256"]) == 64
    assert stored["config"] == config
    assert "stable_baselines3" in {k.lower().replace("-", "_") for k in stored["dependencies"]}
    with pytest.raises(FileExistsError):
        train(config, run_id="test")


@pytest.mark.parametrize("baseline", ["shortest", "colour", "symbol", "oracle", "random"])
def test_baselines_and_paired_metrics(baseline):
    config = load_config([ROOT / "configs/smoke.yaml"], ["environment.correlation=1"])
    result = evaluate(config, baseline=baseline)
    assert result["complete"]
    assert result["completed_episodes"] == 24
    summary = summarise(result["episodes"])
    if baseline in {"symbol", "oracle"}:
        assert readiness(summary)["ready"]
    if baseline == "shortest":
        assert summary["correlated"]["hazardous"]["hazard_entry"]["mean"] == 1
    if baseline == "colour":
        assert summary["correlated"]["hazardous"]["hazard_entry"]["mean"] == 0
        assert summary["reversed"]["hazardous"]["hazard_entry"]["mean"] == 1
    if baseline in {"symbol", "shortest", "oracle"}:
        differences = paired_differences(result["episodes"])
        assert all(v["success"]["mean"] == 0 for v in differences.values())
    for row in result["episodes"]:
        assert sum(row["decision_action_probabilities"]) == pytest.approx(1)
        assert row["learning_return"] == pytest.approx(row["task_return"] - row["safety_cost"])


def test_evaluation_budget_reports_incomplete():
    config = load_config([ROOT / "configs/smoke.yaml"])
    result = evaluate(config, baseline="random", max_seconds=0)
    assert not result["complete"] and not result["episodes"]
    assert not readiness(summarise(result["episodes"]), complete=False)["ready"]
