"""Tests for plotting utilities — TikZ export and matplotlib quick-look plots."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.plotting.tikz_export import TikZExporter
from src.plotting.matplotlib_plots import plot_convergence, plot_bar_comparison, plot_heatmap
from src.plotting.palette import COLORS, AGENT_COLORS, AGENT_MARKERS


def _make_convergence_data(n=50):
    eps = list(range(n))
    return {
        "sac": pd.DataFrame({"episode": eps, "mean": np.random.randn(n).cumsum(), "std": np.abs(np.random.randn(n))}),
        "td3": pd.DataFrame({"episode": eps, "mean": np.random.randn(n).cumsum(), "std": np.abs(np.random.randn(n))}),
    }


# ── Palette ───────────────────────────────────────────────────────────────────

def test_palette_has_required_colors():
    for key in ("SupBlue", "SupRed", "SupGreen", "SupOrange", "SupGray"):
        assert key in COLORS
        r, g, b = COLORS[key]
        assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0


def test_agent_colors_keys():
    for agent in ("sac", "td3", "ddpg", "ppo", "dqn"):
        assert agent in AGENT_COLORS
        assert agent in AGENT_MARKERS


# ── TikZExporter ──────────────────────────────────────────────────────────────

def test_tikz_convergence_generates_tex():
    exporter = TikZExporter()
    data = _make_convergence_data()
    tex = exporter.convergence_plot(data)
    assert r"\begin{axis}" in tex
    assert r"\addplot" in tex
    assert "SAC" in tex or "sac" in tex.lower()


def test_tikz_convergence_writes_file(tmp_path):
    exporter = TikZExporter()
    data = _make_convergence_data()
    out = str(tmp_path / "convergence.tex")
    exporter.convergence_plot(data, output_path=out)
    assert Path(out).exists()
    content = Path(out).read_text()
    assert r"\begin{figure}" in content


def test_tikz_bar_chart(tmp_path):
    exporter = TikZExporter()
    data = {"sac": {"reward": 10.0, "cost": 5.0}, "td3": {"reward": 8.0, "cost": 6.0}}
    out = str(tmp_path / "bars.tex")
    tex = exporter.bar_chart(data, output_path=out)
    assert r"\addplot" in tex
    assert Path(out).exists()


def test_tikz_heatmap(tmp_path):
    exporter = TikZExporter()
    df = pd.DataFrame(
        np.random.rand(3, 3),
        index=[0.5, 1.0, 1.5],
        columns=[0.5, 1.0, 1.5],
    )
    out = str(tmp_path / "heatmap.tex")
    tex = exporter.heatmap(df, output_path=out)
    assert "matrix plot" in tex
    assert Path(out).exists()


def test_tikz_significance_table(tmp_path):
    exporter = TikZExporter()
    comparisons = [
        {"name_a": "sac", "name_b": "td3", "p_value": 0.01, "significant_0.05": True,
         "significant_0.01": True, "effect_size": 0.7, "mean_a": 10.0, "mean_b": 7.0},
    ]
    out = str(tmp_path / "sig.tex")
    tex = exporter.significance_table(comparisons, output_path=out)
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex


# ── Matplotlib quick-look ─────────────────────────────────────────────────────

def test_matplotlib_convergence_saves_png(tmp_path):
    data = _make_convergence_data()
    out = str(tmp_path / "conv.png")
    plot_convergence(data, output_path=out)
    assert Path(out).exists()


def test_matplotlib_bar_saves_png(tmp_path):
    data = {"sac": {"reward": 10.0}, "td3": {"reward": 8.0}}
    out = str(tmp_path / "bars.png")
    plot_bar_comparison(data, output_path=out)
    assert Path(out).exists()


def test_matplotlib_heatmap_saves_png(tmp_path):
    df = pd.DataFrame(np.random.rand(4, 4), index=[0.5, 1.0, 1.5, 2.0], columns=[0.5, 1.0, 1.5, 2.0])
    out = str(tmp_path / "heat.png")
    plot_heatmap(df, output_path=out)
    assert Path(out).exists()
