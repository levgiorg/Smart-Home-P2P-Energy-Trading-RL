from __future__ import annotations

"""TikZ/pgfplots export engine for publication-ready figures."""

from pathlib import Path

import pandas as pd

from .palette import AGENT_COLORS, AGENT_MARKERS
from .templates import (
    BAND_LAYER_TEMPLATE,
    BAR_CHART_TEMPLATE,
    CONVERGENCE_TEMPLATE,
    HEATMAP_TEMPLATE,
    PREAMBLE,
    TABLE_TEMPLATE,
)


def _coords(x: list, y: list) -> str:
    return " ".join(f"({xi},{yi:.4f})" for xi, yi in zip(x, y))


class TikZExporter:
    """Generates publication-ready TikZ/pgfplots .tex files."""

    def __init__(
        self,
        agent_colors: dict[str, str] | None = None,
        agent_markers: dict[str, str] | None = None,
    ) -> None:
        self.agent_colors = agent_colors or AGENT_COLORS
        self.agent_markers = agent_markers or AGENT_MARKERS

    def convergence_plot(
        self,
        data: dict[str, pd.DataFrame],
        xlabel: str = "Episode",
        ylabel: str = "Reward",
        caption: str = "Training convergence.",
        label: str = "convergence",
        output_path: str | None = None,
        smooth_window: int = 10,
    ) -> str:
        """Generate convergence plot with confidence bands.

        Args:
            data: {agent_name: DataFrame with columns [episode, mean, std]}
        """
        band_layers = ""
        for agent, df in data.items():
            color = self.agent_colors.get(agent, "SupGray")
            marker = self.agent_markers.get(agent, "o")
            episodes = df["episode"].tolist()
            means = df["mean"].tolist()
            stds = df["std"].tolist() if "std" in df.columns else [0.0] * len(means)
            upper = [(e, m + s) for e, m, s in zip(episodes, means, stds)]
            lower = [(e, m - s) for e, m, s in zip(episodes, means, stds)]

            band_layers += BAND_LAYER_TEMPLATE.format(
                name=agent,
                color=color,
                marker=marker,
                upper=_coords([x[0] for x in upper], [x[1] for x in upper]),
                lower=_coords([x[0] for x in lower], [x[1] for x in lower]),
                mean=_coords(episodes, means),
                legend=agent.upper(),
            )

        tex = CONVERGENCE_TEMPLATE.format(
            xlabel=xlabel,
            ylabel=ylabel,
            band_layers=band_layers,
            caption=caption,
            label=label,
        )

        if output_path:
            Path(output_path).write_text(PREAMBLE + tex)
        return tex

    def bar_chart(
        self,
        data: dict[str, dict[str, float]],
        errors: dict[str, dict[str, float]] | None = None,
        xlabel: str = "Metric",
        ylabel: str = "Value",
        caption: str = "Algorithm comparison.",
        label: str = "comparison",
        output_path: str | None = None,
    ) -> str:
        """Grouped bar chart with optional error bars."""
        # Gather all metric names
        metrics = sorted({m for agent_data in data.values() for m in agent_data})
        xcoords = ",".join(metrics)

        bars = ""
        for agent, agent_data in data.items():
            color = self.agent_colors.get(agent, "SupGray")
            if errors and agent in errors:
                coords_str = " ".join(
                    f"({m},{agent_data.get(m, 0):.4f}) +- (0,{errors[agent].get(m, 0):.4f})"
                    for m in metrics
                )
                bars += (
                    f"      \\addplot [{color}, error bars/.cd, y dir=both, y explicit]\n"
                    f"        coordinates {{ {coords_str} }};\n"
                    f"      \\addlegendentry{{{agent.upper()}}};\n"
                )
            else:
                coords_str = " ".join(f"({m},{agent_data.get(m, 0):.4f})" for m in metrics)
                bars += (
                    f"      \\addplot [{color}] coordinates {{ {coords_str} }};\n"
                    f"      \\addlegendentry{{{agent.upper()}}};\n"
                )

        tex = BAR_CHART_TEMPLATE.format(
            xlabel=xlabel,
            ylabel=ylabel,
            xcoords=xcoords,
            bars=bars,
            caption=caption,
            label=label,
        )

        if output_path:
            Path(output_path).write_text(PREAMBLE + tex)
        return tex

    def heatmap(
        self,
        data: pd.DataFrame,
        xlabel: str = "Parameter A",
        ylabel: str = "Parameter B",
        caption: str = "Sensitivity heatmap.",
        label: str = "heatmap",
        output_path: str | None = None,
    ) -> str:
        """2D heatmap for sensitivity analysis."""
        rows, cols = data.index.tolist(), data.columns.tolist()
        table_rows = ""
        for r in rows:
            for c in cols:
                table_rows += f"{c} {r} {data.loc[r, c]:.4f}\n"

        tex = HEATMAP_TEMPLATE.format(
            xlabel=xlabel,
            ylabel=ylabel,
            data=table_rows.strip(),
            caption=caption,
            label=label,
        )

        if output_path:
            Path(output_path).write_text(PREAMBLE + tex)
        return tex

    def significance_table(
        self,
        results: list[dict],
        caption: str = "Pairwise significance tests.",
        label: str = "significance",
        output_path: str | None = None,
    ) -> str:
        """LaTeX booktabs table of pairwise significance test results."""
        columns = "llrrrr"
        header = r"Agent A & Agent B & $p$-value & Effect size & Mean A & Mean B"

        table_rows = ""
        for r in results:
            sig = "**" if r.get("significant_0.01") else ("*" if r.get("significant_0.05") else "")
            table_rows += (
                f"    {r['name_a']} & {r['name_b']} & "
                f"{r['p_value']:.4f}{sig} & "
                f"{r.get('effect_size', r.get('effect_size_r', 0)):.3f} & "
                f"{r['mean_a']:.2f} & "
                f"{r['mean_b']:.2f} \\\\\n"
            )

        tex = TABLE_TEMPLATE.format(
            columns=columns,
            header=header,
            rows=table_rows.rstrip(),
            caption=caption,
            label=label,
        )

        if output_path:
            Path(output_path).write_text(tex)
        return tex
