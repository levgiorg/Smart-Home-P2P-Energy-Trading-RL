"""TikZ/pgfplots template strings for publication-ready figures."""

from __future__ import annotations

PREAMBLE = r"""
\definecolor{SupBlue}{RGB}{65,105,225}
\definecolor{SupRed}{RGB}{220,60,60}
\definecolor{SupGreen}{RGB}{60,160,60}
\definecolor{SupOrange}{RGB}{255,140,0}
\definecolor{SupGray}{RGB}{128,128,128}

\pgfplotsset{
  style_classic/.style={
    width=\linewidth,
    height=0.65\linewidth,
    grid=major,
    grid style={dotted,gray!30},
    tick align=outside,
    legend style={font=\small, draw=none, fill=none},
    label style={font=\small},
    tick label style={font=\footnotesize},
    line width=1.0pt,
  },
  style_subplot/.style={
    style_classic,
    height=0.5\linewidth,
    legend columns=2,
  },
}
"""

CONVERGENCE_TEMPLATE = r"""
\begin{{figure}}[t]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
      style_classic,
      xlabel={{{xlabel}}},
      ylabel={{{ylabel}}},
      legend pos=south east,
    ]
{band_layers}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{{caption}}}
  \label{{fig:{label}}}
\end{{figure}}
"""

BAND_LAYER_TEMPLATE = r"""      % {name} — confidence band
      \addplot [name path={name}_upper, draw=none, forget plot] coordinates {{ {upper} }};
      \addplot [name path={name}_lower, draw=none, forget plot] coordinates {{ {lower} }};
      \addplot [{color}!20, fill between/of={name}_upper and {name}_lower, forget plot] fill between;
      \addplot [{color}, mark={marker}, mark repeat=50] coordinates {{ {mean} }};
      \addlegendentry{{{legend}}};
"""

BAR_CHART_TEMPLATE = r"""
\begin{{figure}}[t]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
      style_classic,
      ybar,
      bar width=8pt,
      xlabel={{{xlabel}}},
      ylabel={{{ylabel}}},
      xtick=data,
      symbolic x coords={{{xcoords}}},
      enlarge x limits=0.15,
      legend pos=north east,
    ]
{bars}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{{caption}}}
  \label{{fig:{label}}}
\end{{figure}}
"""

HEATMAP_TEMPLATE = r"""
\begin{{figure}}[t]
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[
      style_classic,
      xlabel={{{xlabel}}},
      ylabel={{{ylabel}}},
      colormap/hot,
      colorbar,
    ]
      \addplot [matrix plot*, point meta=explicit] table [meta=z] {{ {data} }};
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{{caption}}}
  \label{{fig:{label}}}
\end{{figure}}
"""

TABLE_TEMPLATE = r"""
\begin{{table}}[t]
  \centering
  \caption{{{caption}}}
  \label{{tab:{label}}}
  \begin{{tabular}}{{{columns}}}
    \toprule
    {header} \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
