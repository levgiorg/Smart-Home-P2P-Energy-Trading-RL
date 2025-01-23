import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from enhanced_analyzer import EnhancedRunAnalyzer


class VersionAwareAnalyzer(EnhancedRunAnalyzer):
    def __init__(self, base_path, version):
        super().__init__(base_path)
        self.version = version
        # Filter runs to only include specified version
        self.runs = [run for run in self.runs if f"v{version}" in run]
        
    def load_run_data(self, run_id):
        """Version-aware data loading that handles different state dimensions"""
        run_path = self.base_path / run_id
        print(f"\nLoading data for: {run_id}")
        
        # Load hyperparameters with error handling
        hyper_path = run_path / 'hyperparameters.json'
        try:
            with open(hyper_path, 'r') as f:
                hyperparams = {}  # Simplified for now
        except Exception as e:
            hyperparams = {}
        
        # Load metrics with version-specific handling
        data = {}
        data_path = run_path / 'data'
        if not data_path.exists():
            print(f"Error: Data directory not found at {data_path}")
            return hyperparams, data
            
        # Load and verify the structure of rewards_per_house first
        rewards_file = data_path / 'ddpg__rewards_per_house.pkl'
        if rewards_file.exists():
            try:
                with open(rewards_file, 'rb') as f:
                    rewards_data = pickle.load(f)
                if isinstance(rewards_data, list):
                    # Convert to numpy array for consistent handling
                    rewards_array = np.array(rewards_data)
                    # Verify shape matches expected dimensions
                    if len(rewards_array.shape) == 2:
                        n_episodes, n_houses = rewards_array.shape
                        print(f"Detected {n_houses} houses in rewards data")
                        data['rewards_per_house'] = rewards_array
            except Exception as e:
                print(f"Error loading rewards: {str(e)}")
        
        # Load other metrics with dimension awareness
        for file in os.listdir(data_path):
            if file.endswith('.pkl') and not file.startswith('ddpg__rewards_per_house'):
                try:
                    with open(data_path / file, 'rb') as f:
                        file_data = pickle.load(f)
                        metric_name = file.replace('ddpg__', '').replace('.pkl', '')
                        
                        # Handle different data structures based on metric type
                        if isinstance(file_data, list):
                            if len(file_data) > 0:
                                # Convert to numpy array for consistent handling
                                data_array = np.array(file_data)
                                # Reshape if needed based on version
                                if self.version >= 4:
                                    # Handle new dimension structure
                                    if len(data_array.shape) == 2:
                                        n_episodes, n_dims = data_array.shape
                                        if metric_name == 'grid_prices':
                                            # Grid prices should be 1D
                                            data_array = data_array.reshape(n_episodes)
                                data[metric_name] = data_array
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    
        return hyperparams, data

    def _calculate_composite_score(self, df):
        """Calculate composite score with version-aware normalization"""
        numeric_cols = ['final_avg_reward', 'score', 'max_avg_reward', 
                       'avg_trading_profit', 'final_selling_price_ratio', 
                       'avg_p2p_energy']
        
        df_norm = df[numeric_cols].copy()
        
        # Normalize each metric while handling version-specific scales
        for col in numeric_cols:
            if df[col].max() - df[col].min() != 0:
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                df_norm[col] = 1
        
        # Adjust weights based on version
        if self.version >= 4:
            # Updated weights for new state/action space
            weights = {
                'final_avg_reward': 0.3,
                'score': 0.2,
                'max_avg_reward': 0.0,
                'avg_trading_profit': 0.2,
                'final_selling_price_ratio': 0.15,
                'avg_p2p_energy': 0.15
            }
        else:
            # Original weights for old state/action space
            weights = {
                'final_avg_reward': 0.25,
                'score': 0.25,
                'max_avg_reward': 0.0,
                'avg_trading_profit': 0.2,
                'final_selling_price_ratio': 0.15,
                'avg_p2p_energy': 0.15
            }
        
        return sum(df_norm[col] * weight for col, weight in weights.items())

def main():
    base_path = Path("ml-outputs/7000 episodes : 10 houses")
    version = 4  # Change this to analyze different versions
    
    print(f"Analyzing version {version} runs...")
    analyzer = VersionAwareAnalyzer(base_path, version)
    
    print(f"\nFound {len(analyzer.runs)} runs:")
    for run in analyzer.runs:
        print(f"  {run}")
    
    print("\nStarting analysis...")
    try:
        analyzer.plot_top_runs_detailed()
        analyzer.print_detailed_analysis()
        plt.show()
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()