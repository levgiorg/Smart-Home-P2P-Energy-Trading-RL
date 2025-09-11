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
    cartel_penalty_visualization: Cartel penalty component visualizations
"""

from energy_analysis.visualizations.device_control import plot_battery_management
from energy_analysis.visualizations.energy_metrics import plot_energy_consumption_breakdown
from energy_analysis.visualizations.advanced_plots import (
    plot_temperature_comfort_zone,
    plot_p2p_price_convergence,
    plot_p2p_final_comparison_bar
)

__all__ = [
    'plot_battery_management',
    'plot_energy_consumption_breakdown',
    'plot_temperature_comfort_zone',
    'plot_p2p_price_convergence',
    'plot_p2p_final_comparison_bar'
]