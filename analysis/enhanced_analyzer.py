import os
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from run_analyzer import RunAnalyzer


class EnhancedRunAnalyzer(RunAnalyzer):
    def __init__(self, base_path="ml-outputs"):
        super().__init__(base_path)
        
    def plot_metric_distributions(self):
        """Plot distributions of key metrics across all runs"""
        df = self.analyze_all_runs()
        
        metrics_to_plot = [
            'final_avg_reward', 
            'score',
            'max_avg_reward', 
            'avg_trading_profit',
            'final_selling_price_ratio',
            'avg_p2p_energy'
        ]
        
        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            sns.boxplot(data=df, y=metric, ax=axs[i])
            axs[i].set_title(f'Distribution of {metric}')
            
        plt.tight_layout()
        return fig

    def plot_top_runs_detailed(self, n_top=5):
        """Enhanced version of top runs comparison with moving averages"""
        df = self.analyze_all_runs()
        top_runs = df.head(n_top)
        
        # Change to 3x2 grid to include score
        fig, axs = plt.subplots(3, 2, figsize=(15, 18))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_top))
        
        # Plot rewards with moving average
        for idx, (run_id, color) in enumerate(zip(top_runs['run_id'], colors)):
            _, data = self.load_run_data(run_id)
            rewards = np.mean(data['rewards_per_house'], axis=1)
            
            # Plot raw data with alpha
            axs[0, 0].plot(rewards, alpha=0.3, color=color)
            
            # Plot moving average
            window = 100
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            axs[0, 0].plot(moving_avg, label=f'{run_id} (MA)', color=color, linewidth=2)
            
        axs[0, 0].set_title('Average Rewards per Episode\n(with 100-episode moving average)')
        axs[0, 0].legend()
        
        # Plot selling prices relative to grid prices
        for idx, (run_id, color) in enumerate(zip(top_runs['run_id'], colors)):
            _, data = self.load_run_data(run_id)
            price_ratio = np.array(data['selling_prices']) / np.array(data['grid_prices'])[:, None]
            mean_ratio = np.mean(price_ratio, axis=1)
            
            axs[0, 1].plot(mean_ratio, label=f'{run_id}', color=color)
        axs[0, 1].set_title('Selling Price / Grid Price Ratio')
        axs[0, 1].legend()
        
        # Plot cumulative trading profit
        for idx, (run_id, color) in enumerate(zip(top_runs['run_id'], colors)):
            _, data = self.load_run_data(run_id)
            cumulative_profit = np.cumsum(np.mean(data['trading_profit'], axis=0))
            axs[1, 0].plot(cumulative_profit, label=f'{run_id}', color=color)
        axs[1, 0].set_title('Cumulative Trading Profit')
        axs[1, 0].legend()
        
        # Plot P2P energy traded percentage
        for idx, (run_id, color) in enumerate(zip(top_runs['run_id'], colors)):
            _, data = self.load_run_data(run_id)
            energy_percentage = (np.array(data['energy_bought_p2p']) / 
                            (np.array(data['HVAC_energy_cons']) + 1e-6)) * 100
            mean_percentage = np.mean(energy_percentage, axis=0)
            
            axs[1, 1].plot(mean_percentage, label=f'{run_id}', color=color)
        axs[1, 1].set_title('P2P Energy Trading Percentage')
        axs[1, 1].set_ylabel('% of Total Energy')
        axs[1, 1].legend()
        
        # Add new plot for score
        for idx, (run_id, color) in enumerate(zip(top_runs['run_id'], colors)):
            _, data = self.load_run_data(run_id)
            if 'score' in data:
                # Convert score data to numpy array and ensure it's 1D
                scores = np.array(data['score'])
                if scores.ndim > 1:
                    scores = np.mean(scores, axis=1)  # Take mean if it's multi-dimensional
                
                # Plot raw data with alpha
                axs[2, 0].plot(scores, alpha=0.3, color=color)
                
                # Plot moving average
                window = 100
                moving_avg_score = pd.Series(scores).rolling(window=window).mean()
                axs[2, 0].plot(moving_avg_score, label=f'{run_id} (MA)', color=color, linewidth=2)
        
        axs[2, 0].set_title('Score Evolution\n(with 100-episode moving average)')
        axs[2, 0].legend()
        
        # Hide the unused subplot
        axs[2, 1].set_visible(False)
        
        plt.tight_layout()
        return fig

    def print_detailed_analysis(self):
        """Print detailed analysis of all runs"""
        df = self.analyze_all_runs()
        
        print("\n=== Detailed Analysis of All Runs ===\n")
        
        # Print summary statistics
        print("Summary Statistics:")
        print(df[['final_avg_reward', 'score', 'max_avg_reward', 'avg_trading_profit', 
                 'final_selling_price_ratio', 'avg_p2p_energy']].describe())
        
        print("\n=== Top 5 Runs Analysis ===\n")
        top_runs = df.head()
        
        for _, row in top_runs.iterrows():
            print(f"\nRun {row['run_id']}:")
            print(f"Composite Score: {row['composite_score']:.4f}")
            print(f"Score: {row['score']:.4f}")  # Added score printing
            print(f"Final Average Reward: {row['final_avg_reward']:.2f}")
            print(f"Maximum Average Reward: {row['max_avg_reward']:.2f}")
            print(f"Average Trading Profit: {row['avg_trading_profit']:.2f}")
            print(f"Final Price Ratio: {row['final_selling_price_ratio']:.2f}")
            print(f"Average P2P Energy: {row['avg_p2p_energy']:.2f}")
            print(f"Convergence Speed: {row['convergence_speed']} episodes")
            
        return df

 