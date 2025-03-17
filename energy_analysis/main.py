"""
Main script for energy mechanism analysis and visualization.

This script orchestrates the entire analysis process:
1. Classifies runs by mechanism
2. Loads and processes data
3. Generates visualizations
4. Handles errors gracefully
"""


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
import numpy as np
from energy_analysis.config import PLOTS_OUTPUT_DIR
from energy_analysis.utils import classify_runs_by_mechanism
from energy_analysis.data_processor import load_data
from energy_analysis.visualizations import (
    plot_mechanism_comparison,
    plot_temperature_control,
    plot_battery_management,
    plot_energy_consumption_breakdown,
    plot_per_house_performance,
    plot_comparative_matrix,
    plot_box_plots,
    plot_hyperparameter_sensitivity,
    plot_beta_grid_fee_analysis
)

def identify_outliers(data_by_mechanism, threshold=5.0):
    """
    Identify runs with extreme values that may be outliers.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        threshold (float): Threshold value for price ratio outlier detection
        
    Returns:
        dict: Dictionary mapping mechanism types to lists of outlier run IDs
    """
    outliers = {mechanism: [] for mechanism in data_by_mechanism.keys()}
    
    for mechanism, data in data_by_mechanism.items():
        print(f"\nAnalyzing {mechanism} mechanism:")
        
        # Check price ratios
        for i, ratios in enumerate(data['price_ratios']):
            if len(ratios) >= 100:
                avg_ratio = np.mean(ratios[-100:])
                
                # Find the corresponding run_id
                run_id = None
                for hyper_data in data['hyperparameters']:
                    if hyper_data['run_id'] == i + 1:  # Add 1 to match original run IDs
                        run_id = hyper_data['run_id']
                        break
                
                print(f"  Run {run_id}: Average Price Ratio = {avg_ratio:.2f}")
                
                # Check if this is an outlier
                if avg_ratio > threshold:
                    print(f"  *** OUTLIER DETECTED: Run {run_id} with Price Ratio = {avg_ratio:.2f} ***")
                    outliers[mechanism].append(run_id)
    
    return outliers

def generate_plots(data_by_mechanism):
    """
    Generate all plots using the loaded data.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        dict: Dictionary mapping plot names to their file paths
    """
    plot_paths = {}
    plot_functions = [
        ('mechanism_comparison', plot_mechanism_comparison),
        ('temperature_control', plot_temperature_control),
        ('battery_management', plot_battery_management),
        ('energy_consumption', plot_energy_consumption_breakdown),
        ('per_house_performance', plot_per_house_performance),
        ('comparative_matrix', plot_comparative_matrix),
        ('box_plots', plot_box_plots),
        ('metric_sensitivity', plot_hyperparameter_sensitivity),
        ('parameter_impact', plot_beta_grid_fee_analysis)
    ]
    
    for plot_name, plot_function in plot_functions:
        try:
            print(f"Generating {plot_name} visualization...")
            result = plot_function(data_by_mechanism)
            if result:
                if isinstance(result, list):
                    plot_paths[plot_name] = result
                else:
                    plot_paths[plot_name] = [result]
                print(f"✓ {plot_name} generated successfully")
            else:
                print(f"✗ {plot_name} generation failed (no valid data)")
        except Exception as e:
            print(f"✗ Error generating {plot_name}: {e}")
            traceback.print_exc()
    
    return plot_paths


def main():
    """
    Main function to orchestrate the energy mechanism analysis process.
    """
    # Banner
    print("=" * 80)
    print("Energy Mechanism Analysis Tool")
    print("=" * 80)
    
    # Classify runs by mechanism type
    print("\nClassifying runs by mechanism type...")
    runs_by_mechanism = classify_runs_by_mechanism()
    
    # Print summary
    for mechanism, run_ids in runs_by_mechanism.items():
        print(f"{mechanism}: {len(run_ids)} runs - {run_ids}")
    
    # Load data from all runs
    print("\nLoading data from all runs...")
    data_by_mechanism = load_data(runs_by_mechanism)
    
    # Identify outliers
    print("\nScanning for outlier runs...")
    outliers = identify_outliers(data_by_mechanism)

    # Print summary of outliers
    print("\nSummary of detected outliers:")
    for mechanism, outlier_runs in outliers.items():
        if outlier_runs:
            print(f"{mechanism}: {outlier_runs}")
        else:
            print(f"{mechanism}: No outliers detected")

    # Generate all plots
    print("\nGenerating IEEE-compliant plots...")
    plot_paths = generate_plots(data_by_mechanism)
    
    # Final report
    print("\n" + "=" * 80)
    print(f"Analysis complete. {len(plot_paths)} plot types generated.")
    print(f"Plots are saved in the '{PLOTS_OUTPUT_DIR}' directory.")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)