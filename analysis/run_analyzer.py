import os
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

class RunAnalyzer:
    def __init__(self, base_path="ml-outputs"):
        self.base_path = Path(base_path)
        self.runs = [d for d in os.listdir(base_path) if d.startswith('run_')]
        self.metrics = {}
        
    def load_run_data(self, run_id):
        """Load all relevant data for a specific run"""
        run_path = self.base_path / run_id
        
        # Load hyperparameters
        with open(run_path / 'hyperparameters.json', 'r') as f:
            hyperparams = json.load(f)
            
        # Load key metrics
        data = {}
        data_path = run_path / 'data'
        for file in os.listdir(data_path):
            if file.endswith('.pkl'):
                metric_name = file.replace('ddpg__', '').replace('.pkl', '')
                with open(data_path / file, 'rb') as f:
                    data[metric_name] = pickle.load(f)
                    
        return hyperparams, data
    
    def analyze_run(self, run_id):
        """Analyze key metrics for a single run"""
        hyperparams, data = self.load_run_data(run_id)
        
        # Calculate key performance indicators
        metrics = {
            'run_id': run_id,
            'final_avg_reward': np.mean(data['rewards_per_house'][-100:]),  # Average reward in last 100 episodes
            'max_avg_reward': np.max(np.mean(data['rewards_per_house'], axis=1)),  # Maximum average reward
            'avg_trading_profit': np.mean(data['trading_profit']),  # Average trading profit
            'final_selling_price_ratio': np.mean(data['selling_prices'][-100:]) / np.mean(data['grid_prices'][-100:]),  # Final price ratio
            'avg_p2p_energy': np.mean(data['energy_bought_p2p']),  # Average P2P energy traded
            'convergence_speed': self._calculate_convergence(data['rewards_per_house']),  # Episodes to converge
            'score': np.mean(np.array(data['score']).flatten()) if 'score' in data else 0,  
            'hyperparameters': hyperparams
        }
        
        self.metrics[run_id] = metrics
        return metrics
    
    def _calculate_convergence(self, rewards, window=100, threshold=0.05):
        """Calculate number of episodes until convergence"""
        rolling_mean = pd.DataFrame(rewards).rolling(window=window, min_periods=1).mean()
        rolling_std = pd.DataFrame(rewards).rolling(window=window, min_periods=1).std()
        
        # Find where standard deviation becomes small relative to mean
        convergence_point = np.where(rolling_std.mean(axis=1) / np.abs(rolling_mean.mean(axis=1)) < threshold)[0]
        return convergence_point[0] if len(convergence_point) > 0 else len(rewards)
    
    def analyze_all_runs(self):
        """Analyze all runs and rank them based on key metrics"""
        for run_id in self.runs:
            self.analyze_run(run_id)
            
        # Create DataFrame with all metrics
        df = pd.DataFrame([self.metrics[run_id] for run_id in self.runs])
        
        # Normalize and combine metrics for ranking
        df_norm = df.select_dtypes(include=[np.number]).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        # Calculate composite score (you can adjust weights based on priorities)
        weights = {
            'final_avg_reward': 0.3,      # 30% weight
            'score': 0.2,                 # 20% weight
            'max_avg_reward': 0,       
            'avg_trading_profit': 0.2,    # 20% weight
            'final_selling_price_ratio': 0.15,  # 15% weight
            'avg_p2p_energy': 0.15        # 15% weight
        }
        
        df['composite_score'] = sum(df_norm[col] * weight for col, weight in weights.items())
        
        return df.sort_values('composite_score', ascending=False)
    
