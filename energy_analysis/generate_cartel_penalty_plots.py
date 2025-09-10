#!/usr/bin/env python
"""
Script to generate IEEE plots for cartel penalty mechanism visualizations.
"""
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from energy_analysis.config import configure_matplotlib
from energy_analysis.utils import classify_runs_by_mechanism
from energy_analysis.data_processor import load_data
from energy_analysis.visualizations.cartel_penalty_visualization import (
    plot_penalty_components_stacked_area,
    plot_penalty_components_waterfall
)

def main():
    """
    Generate IEEE-style visualizations for cartel penalty components.
    """
    print("Generating cartel penalty visualizations for IEEE Smart Grid journal...")
    
    # Make sure matplotlib is properly configured for IEEE
    configure_matplotlib()
    
    # Get all runs grouped by mechanism type
    runs_by_mechanism = classify_runs_by_mechanism(use_sampling=False)
    
    # Load data for all mechanisms
    data_by_mechanism = load_data(runs_by_mechanism)
    
    # Generate stacked area chart visualization
    print("\nGenerating stacked area chart of penalty components...")
    stacked_area_path = plot_penalty_components_stacked_area(data_by_mechanism)
    if stacked_area_path:
        print(f"Stacked area chart saved to: {stacked_area_path}")
    else:
        print("Failed to generate stacked area chart.")
    
    # Generate waterfall chart visualization
    print("\nGenerating waterfall chart of penalty components...")
    waterfall_path = plot_penalty_components_waterfall(data_by_mechanism)
    if waterfall_path:
        print(f"Waterfall chart saved to: {waterfall_path}")
    else:
        print("Failed to generate waterfall chart.")
    
    print("\nVisualization generation complete.")

if __name__ == "__main__":
    main() 