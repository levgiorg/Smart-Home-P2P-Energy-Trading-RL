"""
Main script for energy mechanism analysis and visualization.

This script orchestrates the entire analysis process:
1. Classifies runs by mechanism or compares algorithms
2. Loads and processes data
3. Generates visualizations
4. Handles errors gracefully

Usage:
    python main.py                                    # Default: mechanism comparison
    python main.py --mode algorithm                   # Algorithm comparison mode
    python main.py --mode algorithm --ddpg-dir runs --dqn-dir dqn_runs  # Custom directories
"""


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import traceback
import numpy as np
from energy_analysis.config import PLOTS_OUTPUT_DIR
from energy_analysis.utils import classify_runs_by_mechanism
from energy_analysis.data_processor import load_data
from energy_analysis.visualizations import (
    plot_battery_management,
    plot_energy_consumption_breakdown,
    plot_temperature_comfort_zone,
    plot_p2p_price_convergence,
    plot_p2p_final_comparison_bar
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

def generate_plots(data_by_mechanism, comparison_mode="mechanism", ddpg_runs_dir="runs", dqn_runs_dir="dqn_runs"):
    """
    Generate all plots using the loaded data.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        comparison_mode (str): Either "mechanism" or "algorithm" for comparison type
        ddpg_runs_dir (str): Directory containing DDPG runs (for algorithm mode)
        dqn_runs_dir (str): Directory containing DQN runs (for algorithm mode)
        
    Returns:
        dict: Dictionary mapping plot names to their file paths
    """
    plot_paths = {}
    plot_functions = [
        # Only the 4 plots the user wants to keep
        ('battery_management', plot_battery_management),
        ('energy_consumption', plot_energy_consumption_breakdown), 
        ('temperature_comfort_zone', plot_temperature_comfort_zone),
        ('p2p_price_convergence', plot_p2p_price_convergence)
    ]
    
    # Add the new bar plot for algorithm comparison mode
    if comparison_mode == "algorithm":
        plot_functions.append(('p2p_final_comparison_bar', plot_p2p_final_comparison_bar))
    
    print(f"Generating visualizations in {comparison_mode} comparison mode...")
    
    for plot_name, plot_function in plot_functions:
        try:
            print(f"Generating {plot_name} visualization...")
            if comparison_mode == "algorithm":
                if plot_name == 'p2p_final_comparison_bar':
                    # Special case for bar plot - doesn't need data_by_mechanism
                    result = plot_function(ddpg_runs_dir, dqn_runs_dir)
                else:
                    result = plot_function(data_by_mechanism, comparison_mode, ddpg_runs_dir, dqn_runs_dir)
            else:
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Energy Mechanism Analysis Tool')
    parser.add_argument('--mode', choices=['mechanism', 'algorithm'], default='mechanism',
                       help='Comparison mode: mechanism (default) or algorithm')
    parser.add_argument('--ddpg-dir', default='runs',
                       help='Directory containing DDPG runs (default: runs)')
    parser.add_argument('--dqn-dir', default='dqn_runs',
                       help='Directory containing DQN runs (default: dqn_runs)')
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 80)
    print("Energy Mechanism Analysis Tool")
    print("=" * 80)
    print(f"Mode: {args.mode.title()} comparison")
    if args.mode == 'algorithm':
        print(f"DDPG runs directory: {args.ddpg_dir}")
        print(f"DQN runs directory: {args.dqn_dir}")
    print("=" * 80)
    
    if args.mode == 'algorithm':
        # Algorithm comparison mode - create dummy data structure
        print("\nRunning in algorithm comparison mode...")
        data_by_mechanism = {}  # Empty dict as plots will load data directly
        
    else:
        # Default mechanism comparison mode
        # Classify runs by mechanism type
        print("\nClassifying runs by mechanism type...")
        runs_by_mechanism = classify_runs_by_mechanism()
        
        # Print summary
        for mechanism, run_ids in runs_by_mechanism.items():
            print(f"{mechanism}: {len(run_ids)} runs - {run_ids}")
        
        # Load data from all runs
        print("\nLoading data from all runs...")
        data_by_mechanism = load_data(runs_by_mechanism)
    
    if args.mode == 'mechanism':
        # Only scan for outliers in mechanism mode
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
    plot_paths = generate_plots(data_by_mechanism, args.mode, args.ddpg_dir, args.dqn_dir)
    
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