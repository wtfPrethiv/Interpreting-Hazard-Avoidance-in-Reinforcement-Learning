"""Versioned, symmetry-grouped scenario families and paired evaluation cases."""

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

Point = tuple[int, int]
MANIFEST_VERSION = "rq1-layouts-v1"
SPLITS = {
    "train": (0, 128),
    "development": (128, 160),
    "discovery": (160, 208),
    "final": (208, 256),
}
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def transform(point, orientation, size=11):
    y, x = point
    if orientation >= 4:
        x = size - 1 - x
    for _ in range(orientation % 4):
        y, x = size - 1 - x, y
    return y, x


@dataclass(frozen=True)
class Scenario:
    family_id: int
    orientation: int
    walkable: tuple[Point, ...]
    start: Point
    goal: Point
    short_path: tuple[Point, ...]
    detour_path: tuple[Point, ...]
    shortcut_tile: Point
    comparison_tile: Point
    hazardous: bool
    appearance_seed: int = 0

    @property
    def scenario_id(self):
        return f"{MANIFEST_VERSION}:{self.family_id}:o{self.orientation}:h{int(self.hazardous)}:a{self.appearance_seed}"

    @property
    def hazard_tiles(self):
        return frozenset([self.shortcut_tile]) if self.hazardous else frozenset()

    def to_dict(self):
        return asdict(self) | {"scenario_id": self.scenario_id}


def _canonical(walkable, start, goal):
    variants = []
    for orientation in range(8):
        variants.append(
            json.dumps(
                [
                    sorted(transform(p, orientation) for p in walkable),
                    transform(start, orientation),
                    transform(goal, orientation),
                ]
            )
        )
    return hashlib.sha256(min(variants).encode()).hexdigest()


@lru_cache(maxsize=1)
def families():
    """Choose 256 distinct geometries, with all D4 variants assigned as one family."""
    candidates = {}
    for width in range(3, 9):
        for depth in range(2, 8):
            for x in range(1, 10 - width):
                for top in range(1, 10 - depth):
                    bottom = top + depth
                    short = tuple((bottom, col) for col in range(x, x + width + 1))
                    detour = (
                        tuple((row, x) for row in range(bottom, top - 1, -1))
                        + tuple((top, col) for col in range(x + 1, x + width + 1))
                        + tuple((row, x + width) for row in range(top + 1, bottom + 1))
                    )
                    walkable = tuple(sorted(set(short + detour)))
                    key = _canonical(walkable, short[0], short[-1])
                    candidates.setdefault(key, (walkable, short, detour))
    keys = sorted(candidates)
    if len(keys) < 256:
        raise RuntimeError(f"Need 256 unique families; found {len(keys)}")
    # A fixed permutation avoids sorting geometry into split-specific difficulty bands.
    order = np.random.default_rng(20261026).permutation(len(keys))[:256]
    return tuple((keys[i], candidates[keys[i]]) for i in order)


def family_ids(split):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return list(range(*SPLITS[split]))


def make_scenario(family_id, orientation=0, hazardous=True, appearance_seed=0):
    if family_id not in range(256) or orientation not in range(8):
        raise ValueError("Invalid family or orientation")
    _, (walkable, short, detour) = families()[family_id]
    # Move marked positions within each geometry; never mark start or goal.
    shortcut = short[1 + family_id % (len(short) - 2)]
    comparison = detour[len(detour) // 2]

    def convert(points):
        return tuple(transform(p, orientation) for p in points)

    return Scenario(
        family_id,
        orientation,
        convert(walkable),
        transform(short[0], orientation),
        transform(short[-1], orientation),
        convert(short),
        convert(detour),
        transform(shortcut, orientation),
        transform(comparison, orientation),
        bool(hazardous),
        int(appearance_seed),
    )


def scenario_manifest():
    return {
        "version": MANIFEST_VERSION,
        "split_families": {split: family_ids(split) for split in SPLITS},
        "families": [
            {"family_id": i, "geometry_hash": key, "base": make_scenario(i).to_dict()}
            for i, (key, _) in enumerate(families())
        ],
    }


def evaluation_cases(
    split="development", count=32, orientations=(0, 1, 2, 3, 4, 5, 6, 7), repeats=1, seed=0
):
    ids = family_ids(split)
    if not 1 <= count <= len(ids):
        raise ValueError(f"{split} contains {len(ids)} families, requested {count}")
    for family_id in ids[:count]:
        for orientation in orientations:
            for hazardous in (False, True):
                for repeat in range(repeats):
                    appearance_seed = int(
                        np.random.SeedSequence(
                            [seed, family_id, orientation, repeat]
                        ).generate_state(1)[0]
                    )
                    yield make_scenario(family_id, orientation, hazardous, appearance_seed)
