"""
Configuration settings for the energy analysis package.
Contains constants, paths, and matplotlib configuration.
"""
import os
import matplotlib as mpl

# Directory configuration
ML_OUTPUT_DIR = "ml-outputs2"
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


# Configure matplotlib for IEEE-compliant plots
def configure_matplotlib():
    """Configure matplotlib settings for IEEE-compliant figures."""
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['axes.axisbelow'] = True  # grid lines behind data
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    mpl.rcParams['pdf.fonttype'] = 42

# Run configuration setup
configure_matplotlib()