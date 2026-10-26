"""Static research figures from recorded behavioural results only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def behavioural_figures(paths, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [json.loads(Path(path).read_text()) for path in paths]
    if not datasets or any(not item["complete"] for item in datasets):
        raise ValueError("Figures require complete evaluations")
    conditions = ["correlated", "randomised", "reversed"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        ("hazardous", "success", "Goal success on hazardous layouts"),
        ("hazardous", "hazard_entry", "Any hazard entry"),
        ("safe", "short_route_success", "Successful safe-shortcut use"),
    ]
    for data in datasets:
        label = (
            data["baseline"]
            or f"{data['config']['environment']['colour_mode']} PPO (seed {data['config']['seed']})"
        )
        for axis, (group, metric, title) in zip(axes, panels):
            available = [condition for condition in conditions if condition in data["summary"]]
            stats = [data["summary"][condition][group][metric] for condition in available]
            means = np.array([row["mean"] for row in stats])
            intervals = np.array([row["ci95"] for row in stats])
            error = np.maximum(0, np.stack([means - intervals[:, 0], intervals[:, 1] - means]))
            axis.errorbar(
                range(len(available)), means, yerr=error, marker="o", capsize=3, label=label
            )
            axis.set(
                title=title,
                ylim=(-0.03, 1.03),
                xticks=range(len(available)),
                xticklabels=available,
                ylabel="Proportion of episodes",
            )
            axis.tick_params(axis="x", rotation=15)
            axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    label = (
        "smoke checks"
        if all(data["kind"] == "smoke" for data in datasets)
        else "development pilots"
    )
    fig.suptitle(f"RQ1 {label} — intervals resample layout families, not training seeds")
    destination = output_dir / "rq1_behavioural_pilots.png"
    fig.savefig(destination, dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)
    return str(destination)
