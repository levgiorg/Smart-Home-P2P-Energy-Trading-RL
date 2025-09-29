"""
Configuration settings for the energy analysis package.
Contains constants, paths, and matplotlib configuration.
"""
import os
import matplotlib as mpl

# Directory configuration
ML_OUTPUT_DIR = "runs"
PLOTS_OUTPUT_DIR = "energy_analysis/ieee_plots"

# Create output directory if it doesn't exist
if not os.path.exists(PLOTS_OUTPUT_DIR):
    os.makedirs(PLOTS_OUTPUT_DIR)

# IEEE colors palette
IEEE_COLORS = {
    'blue': '#0072BD',
    'green': '#77AC30',
    'red': '#A2142F',
    'orange': '#EDB120',
    'purple': '#7E2F8E'
}

# Mechanism-related constants
MECHANISMS = ['detection', 'ceiling', 'null']
MECHANISM_COLORS = {
    'detection': IEEE_COLORS['blue'],
    'ceiling': IEEE_COLORS['green'],
    'null': IEEE_COLORS['red']
}

MECHANISM_DISPLAY_NAMES = {
    'detection': 'Reward-Based',
    'ceiling': 'Threshold-Based',
    'null': 'No Control Method'
}


# IEEE Publication Standard Settings
PUBLICATION_SETTINGS = {
    # Font sizes (IEEE compliant)
    'title_fontsize': 14,
    'axis_label_fontsize': 12,
    'tick_label_fontsize': 10,
    'legend_fontsize': 10,

    # Figure sizes
    'single_plot_size': (7, 5),        # Standard single plots
    'bar_chart_size': (10, 6),         # Bar charts
    'multi_panel_size': (12, 4),       # Multi-panel plots
    'small_plot_size': (7, 5),         # Previously small plots (upgraded)

    # Line and visual properties
    'standard_dpi': 600,
    'line_width': 2.0,
    'grid_line_width': 1.0,
    'grid_alpha': 0.3,
    'legend_framealpha': 0.9,
    'legend_edgecolor': 'lightgray',

    # Border properties
    'spine_linewidth': 1.5,
    'spine_color': 'black'
}

def configure_matplotlib():
    """Configure matplotlib settings for IEEE-compliant figures."""
    # Use fallback fonts if Times New Roman is not available
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    mpl.rcParams['axes.labelsize'] = PUBLICATION_SETTINGS['axis_label_fontsize']
    mpl.rcParams['axes.titlesize'] = PUBLICATION_SETTINGS['title_fontsize']
    mpl.rcParams['xtick.labelsize'] = PUBLICATION_SETTINGS['tick_label_fontsize']
    mpl.rcParams['ytick.labelsize'] = PUBLICATION_SETTINGS['tick_label_fontsize']
    mpl.rcParams['legend.fontsize'] = PUBLICATION_SETTINGS['legend_fontsize']
    mpl.rcParams['figure.dpi'] = PUBLICATION_SETTINGS['standard_dpi']

    # Suppress font warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
    mpl.rcParams['savefig.dpi'] = PUBLICATION_SETTINGS['standard_dpi']
    mpl.rcParams['lines.linewidth'] = PUBLICATION_SETTINGS['line_width']
    mpl.rcParams['grid.linewidth'] = PUBLICATION_SETTINGS['grid_line_width']
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = PUBLICATION_SETTINGS['grid_alpha']
    mpl.rcParams['axes.axisbelow'] = True  # grid lines behind data
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    mpl.rcParams['pdf.fonttype'] = 42

    # Configure plot borders
    mpl.rcParams['axes.linewidth'] = PUBLICATION_SETTINGS['spine_linewidth']
    mpl.rcParams['axes.edgecolor'] = PUBLICATION_SETTINGS['spine_color']

def apply_publication_style(ax, add_borders=True):
    """
    Apply standardized publication styling to a matplotlib axis.

    Args:
        ax: matplotlib axis object
        add_borders: Whether to add black borders around the plot
    """
    if add_borders:
        # Add black borders around the plot
        for spine in ax.spines.values():
            spine.set_linewidth(PUBLICATION_SETTINGS['spine_linewidth'])
            spine.set_color(PUBLICATION_SETTINGS['spine_color'])
            spine.set_visible(True)

    # Configure tick parameters
    ax.tick_params(axis='both', labelsize=PUBLICATION_SETTINGS['tick_label_fontsize'])

    # Configure grid
    ax.grid(True, alpha=PUBLICATION_SETTINGS['grid_alpha'],
            linewidth=PUBLICATION_SETTINGS['grid_line_width'])
    ax.set_axisbelow(True)

# Run configuration setup
configure_matplotlib()