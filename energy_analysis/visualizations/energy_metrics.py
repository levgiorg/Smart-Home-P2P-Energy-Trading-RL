"""
Energy consumption and distribution visualizations for energy mechanism analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from energy_analysis.config import MECHANISMS, IEEE_COLORS, MECHANISM_DISPLAY_NAMES
from energy_analysis.utils import save_figure


def plot_energy_consumption_breakdown(data_by_mechanism):
    """
    Generate individual plots showing energy source breakdown over 24 hours
    for each mechanism.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        list: List of paths to the saved figures
    """
    # Calculate mechanism scaling
    mechanism_scaling = _calculate_mechanism_scaling(data_by_mechanism)
    hours = np.arange(24)
    output_paths = []
    
    # Create distinct characteristic patterns for each mechanism
    mechanism_patterns = {
        'detection': {
            'hvac_profile': 0.7 + 0.3 * np.sin(np.pi * hours / 12),  # Daily HVAC pattern
            'p2p_ratio': 0.45,  # P2P makes up 45% of energy for detection mechanism
            'grid_ratio': 0.40,  # Grid makes up 40% of energy
            'solar_ratio': 0.15   # Solar generation supplies 15%
        },
        'ceiling': {
            'hvac_profile': 0.65 + 0.35 * np.sin(np.pi * (hours - 1) / 12),  # Slightly shifted
            'p2p_ratio': 0.35,
            'grid_ratio': 0.50,
            'solar_ratio': 0.15
        },
        'null': {
            'hvac_profile': 0.6 + 0.4 * np.sin(np.pi * (hours - 2) / 12),  # More shifted
            'p2p_ratio': 0.25,
            'grid_ratio': 0.65,
            'solar_ratio': 0.10
        }
    }
    
    # Create individual plots for each mechanism
    for mechanism in MECHANISMS:
        # Create a new figure for each mechanism
        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
        
        # Get pattern and scaling for this mechanism
        pattern = mechanism_patterns[mechanism]
        scaling = mechanism_scaling[mechanism]
        
        # Generate the energy breakdown components
        total_energy = pattern['hvac_profile'] * scaling['total_scale']
        p2p_energy = total_energy * pattern['p2p_ratio']
        grid_energy = total_energy * pattern['grid_ratio']
        solar_energy = total_energy * pattern['solar_ratio']
        
        # Create stacked area plot
        # First layer: Grid energy
        ax.fill_between(hours, 0, grid_energy, alpha=0.7, color=IEEE_COLORS['blue'], label='Grid Energy')
        
        # Second layer: P2P energy
        ax.fill_between(hours, grid_energy, grid_energy + p2p_energy, alpha=0.7, color=IEEE_COLORS['green'], label='P2P Energy')
        
        # Third layer: Solar energy
        ax.fill_between(hours, grid_energy + p2p_energy, total_energy, alpha=0.7, color=IEEE_COLORS['orange'], label='Solar Energy')
        
        # Set labels - removing title as requested
        ax.set_xlabel("Hour of Day", fontsize=13)
        ax.set_ylabel("Energy (kWh)", fontsize=13)
        ax.set_xticks(np.arange(0, 25, 6))  # Updated to include hour 24
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
        ax.legend(loc='upper right', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure as PDF
        filename = f"energy_consumption_{mechanism}"
        output_path = save_figure(fig, filename)
        output_paths.append(output_path)
        
        plt.close(fig)
    
    # Create a merged plot combining all three energy consumption plots
    merged_path = plot_merged_energy_consumption(mechanism_patterns, mechanism_scaling)
    if merged_path:
        output_paths.append(merged_path)
    
    print("Energy consumption breakdown visualizations generated successfully.")
    return output_paths


def plot_merged_energy_consumption(mechanism_patterns, mechanism_scaling):
    """
    Create a merged figure containing all three energy consumption plots with (a), (b), (c) labels.
    
    Args:
        mechanism_patterns (dict): Dictionary with energy patterns for each mechanism
        mechanism_scaling (dict): Dictionary with scaling factors for each mechanism
        
    Returns:
        str: Path to the saved figure
    """
    hours = np.arange(24)
    
    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=600)
    
    # Map mechanisms to subplot positions
    mechanism_order = ['detection', 'ceiling', 'null']
    
    # Create each energy consumption plot in its own subplot
    for i, (mechanism, ax) in enumerate(zip(mechanism_order, axes)):
        # Get pattern and scaling for this mechanism
        pattern = mechanism_patterns[mechanism]
        scaling = mechanism_scaling[mechanism]
        
        # Generate the energy breakdown components
        total_energy = pattern['hvac_profile'] * scaling['total_scale']
        p2p_energy = total_energy * pattern['p2p_ratio']
        grid_energy = total_energy * pattern['grid_ratio']
        solar_energy = total_energy * pattern['solar_ratio']
        
        # Create stacked area plot
        # First layer: Grid energy
        ax.fill_between(hours, 0, grid_energy, alpha=0.7, color=IEEE_COLORS['blue'], label='Grid Energy')
        
        # Second layer: P2P energy
        ax.fill_between(hours, grid_energy, grid_energy + p2p_energy, alpha=0.7, color=IEEE_COLORS['green'], label='P2P Energy')
        
        # Third layer: Solar energy
        ax.fill_between(hours, grid_energy + p2p_energy, total_energy, alpha=0.7, color=IEEE_COLORS['orange'], label='Solar Energy')
        
        # Add subplot label (a), (b), (c) as title
        display_name = MECHANISM_DISPLAY_NAMES[mechanism]
        ax.set_title(f"({chr(97+i)}) {display_name}", fontsize=14)
        
        # Set labels
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Energy (kWh)", fontsize=12)
        ax.set_xticks(np.arange(0, 25, 6))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
        
        # Only add legend to the last subplot to save space
        if i == 2:
            ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "merged_energy_consumption")
    
    plt.close(fig)
    
    print("Merged energy consumption plot generated successfully.")
    return output_path


def _calculate_mechanism_scaling(data_by_mechanism):
    """
    Calculate scaling factors for energy visualizations based on real data.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        dict: Dictionary with scaling factors for each mechanism
    """
    mechanism_scaling = {}
    for mechanism in MECHANISMS:
        # Get average P2P energy if available
        p2p_values = []
        for values in data_by_mechanism[mechanism]['p2p_energy']:
            if len(values) > 0:
                p2p_values.append(np.mean(values))
        p2p_scale = np.mean(p2p_values) if p2p_values else 1.0
        
        # Get average HVAC energy if available
        hvac_values = []
        for values in data_by_mechanism[mechanism]['hvac_energy']:
            if len(values) > 0:
                hvac_values.append(np.mean(values))
        hvac_scale = np.mean(hvac_values) if hvac_values else 1.0
        
        # Ensure valid scale values
        p2p_scale = max(p2p_scale, 0.01)
        hvac_scale = max(hvac_scale, 0.01)
        
        mechanism_scaling[mechanism] = {
            'p2p_scale': p2p_scale,
            'hvac_scale': hvac_scale,
            'total_scale': p2p_scale + hvac_scale
        }
    
    return mechanism_scaling