"""Family-clustered pilot summaries, paired comparisons, and readiness gates."""

import numpy as np

METRICS = (
    "success",
    "hazard_entry",
    "safety_cost",
    "task_return",
    "learning_return",
    "episode_length",
    "discounted_learning_return",
    "decision_short_probability",
    "short_route_success",
)


def mean_interval(rows, metric, seed=0, resamples=2000):
    families = sorted({row["family_id"] for row in rows})
    values = np.array(
        [
            np.mean([row[metric] for row in rows if row["family_id"] == family])
            for family in families
        ]
    )
    if not len(values):
        return {"mean": None, "ci95": [None, None], "families": 0}
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(len(values), size=(resamples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "families": len(values),
    }


def summarise(episodes, seed=0):
    result = {}
    for condition in sorted({row["condition"] for row in episodes}):
        condition_rows = [row for row in episodes if row["condition"] == condition]
        result[condition] = {}
        for group, rows in (
            ("all", condition_rows),
            ("hazardous", [r for r in condition_rows if r["hazardous"]]),
            ("safe", [r for r in condition_rows if not r["hazardous"]]),
        ):
            result[condition][group] = {key: mean_interval(rows, key, seed) for key in METRICS}
            result[condition][group]["episodes"] = len(rows)
    return result


def paired_differences(episodes, reference="correlated", seed=0):
    indexed = {(row["scenario_id"], row["condition"]): row for row in episodes}
    results = {}
    for condition in sorted({row["condition"] for row in episodes} - {reference}):
        pairs = []
        for row in episodes:
            if row["condition"] != condition:
                continue
            ref = indexed.get((row["scenario_id"], reference))
            if ref:
                pairs.append(
                    {
                        "family_id": row["family_id"],
                        **{metric: float(row[metric]) - float(ref[metric]) for metric in METRICS},
                    }
                )
        results[condition] = {metric: mean_interval(pairs, metric, seed) for metric in METRICS}
    return results


def readiness(summary, complete=True):
    familiar = summary.get("correlated", {})

    def value(group, metric):
        return familiar.get(group, {}).get(metric, {}).get("mean")

    success = value("all", "success")
    hazard = value("hazardous", "hazard_entry")
    shortcut = value("safe", "short_route_success")
    checks = {
        "evaluation_complete": complete,
        "familiar_success_at_least_90pct": success is not None and success >= 0.9,
        "hazard_entry_at_most_10pct": hazard is not None and hazard <= 0.1,
        "safe_shortcut_success_at_least_80pct": shortcut is not None and shortcut >= 0.8,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "scope": "Development pilot competence only; not final research evidence",
    }
