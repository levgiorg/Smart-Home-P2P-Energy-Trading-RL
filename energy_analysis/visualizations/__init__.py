"""
Visualization subpackage for energy mechanism analysis.

This subpackage contains various visualization modules for different aspects
of the energy mechanism analysis:

Modules:
    core_metrics: Core performance metrics visualizations
    energy_metrics: Energy consumption and distribution visualizations
    device_control: Temperature and battery control visualizations
    statistical: Statistical and comparative visualizations
    sensitivity: Parameter sensitivity analysis visualizations
    advanced_plots: Advanced story-telling visualizations for publication
"""

from energy_analysis.visualizations.core_metrics import plot_mechanism_comparison
from energy_analysis.visualizations.device_control import (
    plot_temperature_control,
    plot_battery_management
)
from energy_analysis.visualizations.energy_metrics import (
    plot_energy_consumption_breakdown
)
from energy_analysis.visualizations.statistical import (
    plot_per_house_performance,
    plot_comparative_matrix,
    plot_box_plots
)
from energy_analysis.visualizations.sensitivity import (
    plot_hyperparameter_sensitivity,
    plot_beta_grid_fee_analysis
)
from energy_analysis.visualizations.advanced_plots import (
    plot_energy_sankey,
    plot_temperature_comfort_zone,
    plot_24h_energy_price_correlation,
    plot_unified_mechanism_comparison,
    plot_market_dynamics_evolution,
    plot_radar_mechanism_comparison,
    plot_daily_energy_flow_diagram
)

__all__ = [
    'plot_mechanism_comparison',
    'plot_temperature_control',
    'plot_battery_management',
    'plot_energy_consumption_breakdown',
    'plot_per_house_performance',
    'plot_comparative_matrix',
    'plot_box_plots',
    'plot_hyperparameter_sensitivity',
    'plot_beta_grid_fee_analysis',
    'plot_energy_sankey',
    'plot_temperature_comfort_zone',
    'plot_24h_energy_price_correlation',
    'plot_unified_mechanism_comparison',
    'plot_market_dynamics_evolution',
    'plot_radar_mechanism_comparison',
    'plot_daily_energy_flow_diagram'
]