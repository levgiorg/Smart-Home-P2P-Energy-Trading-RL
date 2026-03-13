"""Color palette shared by matplotlib and TikZ exports.

SupBlue is always assigned to the best-performing method.
Assignment can be overridden by passing agent_colors to plot functions.
"""

from __future__ import annotations

# Matplotlib RGB tuples (0–1 normalized)
COLORS: dict[str, tuple[float, float, float]] = {
    "SupBlue": (65 / 255, 105 / 255, 225 / 255),
    "SupRed": (220 / 255, 60 / 255, 60 / 255),
    "SupGreen": (60 / 255, 160 / 255, 60 / 255),
    "SupOrange": (255 / 255, 140 / 255, 0 / 255),
    "SupGray": (128 / 255, 128 / 255, 128 / 255),
    "black": (0.0, 0.0, 0.0),
}

# TikZ color definitions (RGB 0–255)
TIKZ_COLORS: dict[str, str] = {
    "SupBlue": r"\definecolor{SupBlue}{RGB}{65,105,225}",
    "SupRed": r"\definecolor{SupRed}{RGB}{220,60,60}",
    "SupGreen": r"\definecolor{SupGreen}{RGB}{60,160,60}",
    "SupOrange": r"\definecolor{SupOrange}{RGB}{255,140,0}",
    "SupGray": r"\definecolor{SupGray}{RGB}{128,128,128}",
}

# Default agent → color mapping (overridable after experiments by performance)
AGENT_COLORS: dict[str, str] = {
    "sac": "SupBlue",
    "td3": "SupRed",
    "ddpg": "SupGreen",
    "ppo": "black",
    "dqn": "SupGray",
}

AGENT_MARKERS: dict[str, str] = {
    "sac": "s",   # square
    "td3": "^",   # triangle
    "ddpg": "D",  # diamond
    "ppo": "o",   # circle
    "dqn": "x",   # x
}
