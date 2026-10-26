"""Pure RGB rendering. Appearance never determines simulator consequences."""

import numpy as np

RED = np.array([220, 60, 65], dtype=np.uint8)
BLUE = np.array([55, 100, 225], dtype=np.uint8)


def tile_colours(scenario, mode="correlated", correlation=0.95):
    if mode not in {"correlated", "randomised", "reversed"}:
        raise ValueError(f"Unknown colour mode {mode}")
    rng = np.random.default_rng(scenario.appearance_seed)
    colours = {}
    for point, hazard in (
        (scenario.shortcut_tile, scenario.hazardous),
        (scenario.comparison_tile, False),
    ):
        if mode == "randomised":
            red = bool(rng.integers(2))
        elif mode == "reversed":
            red = not hazard
        else:
            red = hazard if rng.random() < correlation else not hazard
        colours[point] = RED if red else BLUE
    return colours


def render_observation(scenario, agent_position, colours):
    image = np.full((88, 88, 3), [25, 28, 35], dtype=np.uint8)
    for y, x in scenario.walkable:
        image[y * 8 : (y + 1) * 8, x * 8 : (x + 1) * 8] = [125, 128, 135]
    for point, colour in colours.items():
        y, x = point
        tile = image[y * 8 : (y + 1) * 8, x * 8 : (x + 1) * 8]
        tile[:] = colour
        if point in scenario.hazard_tiles:
            for i in range(2, 6):
                tile[i, i] = 255
                tile[i, 7 - i] = 255
        else:
            for py, px in ((2, 3), (2, 4), (3, 2), (4, 2), (5, 3), (5, 4), (3, 5), (4, 5)):
                tile[py, px] = 255
    y, x = scenario.goal
    image[y * 8 : (y + 1) * 8, x * 8 : (x + 1) * 8] = [50, 205, 75]
    image[y * 8 + 3 : y * 8 + 5, x * 8 + 3 : x * 8 + 5] = 255
    y, x = agent_position
    tile = image[y * 8 : (y + 1) * 8, x * 8 : (x + 1) * 8]
    # Border preserves the status symbol when the agent occupies a marked tile.
    tile[0, :] = tile[-1, :] = tile[:, 0] = tile[:, -1] = [30, 245, 235]
    return np.ascontiguousarray(image.transpose(2, 0, 1))
