"""
Data loading and processing functions for energy mechanism analysis.
"""
import os
import json
import pickle
import numpy as np
import time
from energy_analysis.config import ML_OUTPUT_DIR


def load_data(runs_by_mechanism):
    """
    Load data from all run folders with proper preprocessing.
    
    Args:
        runs_by_mechanism (dict): Dictionary mapping mechanism types to lists of run IDs
        
    Returns:
        dict: Dictionary containing processed data for each mechanism type
    """
    data_by_mechanism = {
        mechanism_type: {
            'rewards': [],
            'price_ratios': [],
            'trading_profits': [],
            'p2p_energy': [],
            'temperatures': [],
            'battery_data': [],
            'hvac_energy': [],
            'penalty': [],
            'hyperparameters': [],
            'beta_values': [],
            'grid_fees': [],
            'comfort_penalties': [],
            'selling_prices': [],
            'grid_prices': [],
            'anti_cartel_penalties': [],
            'depreciation': []
        } for mechanism_type in runs_by_mechanism.keys()
    }
    
    # Calculate total runs to process
    total_runs = sum(len(runs) for runs in runs_by_mechanism.values())
    processed_runs = 0
    
    start_time = time.time()
    
    for mechanism_type, run_ids in runs_by_mechanism.items():
        print(f"\nProcessing {mechanism_type} mechanism runs ({len(run_ids)} runs):")
        
        for run_id in run_ids:
            processed_runs += 1
            run_dir = os.path.join(ML_OUTPUT_DIR, f"run_{run_id}")
            print(f"  Loading run {run_id}/{total_runs} ({processed_runs/total_runs*100:.1f}% complete) - {run_dir}")
            
            # Check if the directory exists
            if not os.path.exists(run_dir):
                print(f"  Warning: Run directory {run_dir} does not exist. Skipping.")
                continue
            
            # Check if the data directory exists
            data_dir = os.path.join(run_dir, 'data')
            if not os.path.exists(data_dir):
                print(f"  Warning: Data directory {data_dir} does not exist. Skipping.")
                continue
                
            try:
                # Load hyperparameters for this run
                _load_hyperparameters(run_dir, run_id, mechanism_type, data_by_mechanism)
                
                # Load rewards data
                _load_rewards_data(run_dir, run_id, mechanism_type, data_by_mechanism)
                
                # Load price data and calculate ratios
                _load_price_data(run_dir, run_id, mechanism_type, data_by_mechanism)
                
                # Load trading profits
                _load_trading_profits(run_dir, run_id, mechanism_type, data_by_mechanism)
                
                # Load P2P energy trading data
                _load_p2p_energy_data(run_dir, run_id, mechanism_type, data_by_mechanism)
                
                # Load temperature data
                _load_temperature_data(run_dir, run_id, mechanism_type, data_by_mechanism)
                
                # Load battery-related data
                _load_battery_data(run_dir, run_id, mechanism_type, data_by_mechanism)
                
                # Load anti-cartel penalties
                _load_anticartel_penalties(run_dir, run_id, mechanism_type, data_by_mechanism)
                
            except Exception as e:
                print(f"  Error processing run {run_dir}: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"\nFinished loading {total_runs} runs in {elapsed_time:.2f} seconds")
    
    # Print summary of loaded data
    print("\nSummary of loaded data:")
    for mechanism_type in data_by_mechanism:
        print(f"  {mechanism_type}:")
        for key, values in data_by_mechanism[mechanism_type].items():
            if isinstance(values, list):
                print(f"    {key}: {len(values)} entries")
    
    return data_by_mechanism


def _load_hyperparameters(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load hyperparameters for a specific run."""
    try:
        hyperparams_path = os.path.join(run_dir, "hyperparameters.json")
        if not os.path.exists(hyperparams_path):
            print(f"    Warning: Hyperparameters file not found at {hyperparams_path}")
            return None, None, None
            
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
            data_by_mechanism[mechanism_type]['hyperparameters'].append({
                'run_id': run_id,
                'params': hyperparams
            })
            
            # Extract hyperparameter values for sensitivity analysis
            if 'reward' in hyperparams and 'beta' in hyperparams['reward']:
                beta = hyperparams['reward']['beta']
            else:
                beta = None
                
            if 'environment' in hyperparams and 'grid_fee' in hyperparams['environment']:
                grid_fee = hyperparams['environment']['grid_fee']
            else:
                grid_fee = None
                
            if 'environment' in hyperparams and 'comfort_penalty' in hyperparams['environment']:
                comfort_penalty = hyperparams['environment']['comfort_penalty']
            elif 'environment' in hyperparams and 'temperature_comfort_penalty_weight' in hyperparams['environment']:
                comfort_penalty = hyperparams['environment']['temperature_comfort_penalty_weight']
            else:
                comfort_penalty = None
            
            return beta, grid_fee, comfort_penalty
    except Exception as e:
        print(f"    Warning: Could not load hyperparameters for {run_dir}: {e}")
        return None, None, None


def _load_rewards_data(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load rewards data for a specific run."""
    try:
        rewards_file = os.path.join(run_dir, "data", "ddpg__rewards_per_house.pkl")
        beta = None
        
        # Try to get beta from hyperparameters
        for hyper_data in data_by_mechanism[mechanism_type]['hyperparameters']:
            if hyper_data['run_id'] == run_id and 'params' in hyper_data:
                if 'reward' in hyper_data['params'] and 'beta' in hyper_data['params']['reward']:
                    beta = hyper_data['params']['reward']['beta']
                    break
        
        if os.path.exists(rewards_file):
            with open(rewards_file, "rb") as f:
                rewards_data = pickle.load(f)
                # Pre-process to ensure right dimensionality (mean across houses if needed)
                if isinstance(rewards_data, np.ndarray) and rewards_data.ndim > 1:
                    rewards_data = np.mean(rewards_data, axis=1)
                data_by_mechanism[mechanism_type]['rewards'].append(rewards_data)
                
                # Store beta value with rewards for analysis
                if beta is not None:
                    # Calculate final reward (last 100 episodes average)
                    if len(rewards_data) >= 100:
                        final_reward = np.mean(rewards_data[-100:])
                        data_by_mechanism[mechanism_type]['beta_values'].append((beta, final_reward))
        else:
            # Try alternate filenames
            alternate_files = [
                os.path.join(run_dir, "data", "ddpg__score.pkl"),
                os.path.join(run_dir, "data", "ddpg__reward.pkl")
            ]
            
            for alt_file in alternate_files:
                if os.path.exists(alt_file):
                    with open(alt_file, "rb") as f:
                        score_data = pickle.load(f)
                        # Handle different data structures
                        if isinstance(score_data, np.ndarray) and score_data.ndim > 1:
                            score_data = np.mean(score_data, axis=1)
                        data_by_mechanism[mechanism_type]['rewards'].append(score_data)
                        
                        # Store beta value with rewards for analysis
                        if beta is not None:
                            # Calculate final reward (last 100 episodes average)
                            if len(score_data) >= 100:
                                final_reward = np.mean(score_data[-100:])
                                data_by_mechanism[mechanism_type]['beta_values'].append((beta, final_reward))
                    break
    except Exception as e:
        print(f"Could not load rewards for {run_dir}: {e}")


def _load_price_data(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load price data and calculate price ratios for a specific run."""
    try:
        prices_file = os.path.join(run_dir, "data", "ddpg__selling_prices.pkl")
        grid_file = os.path.join(run_dir, "data", "ddpg__grid_prices.pkl")
        
        # Get grid_fee from hyperparameters
        grid_fee = None
        for hyper_data in data_by_mechanism[mechanism_type]['hyperparameters']:
            if hyper_data['run_id'] == run_id and 'params' in hyper_data:
                if 'environment' in hyper_data['params'] and 'grid_fee' in hyper_data['params']['environment']:
                    grid_fee = hyper_data['params']['environment']['grid_fee']
                    break
        
        if os.path.exists(prices_file) and os.path.exists(grid_file):
            with open(prices_file, "rb") as f:
                selling_prices = pickle.load(f)
                data_by_mechanism[mechanism_type]['selling_prices'].append(selling_prices)
                
            with open(grid_file, "rb") as f:
                grid_prices = pickle.load(f)
                data_by_mechanism[mechanism_type]['grid_prices'].append(grid_prices)
            
            # Convert to numpy arrays if not already
            selling_prices = np.array(selling_prices)
            grid_prices = np.array(grid_prices)
            
            # Calculate ratio safely with proper dimensionality handling
            if selling_prices.ndim == grid_prices.ndim:
                if selling_prices.ndim > 1:
                    # Mean across houses/agents if needed
                    s_mean = np.mean(selling_prices, axis=1)
                    g_mean = np.mean(grid_prices, axis=1)
                    # Avoid division by zero
                    g_mean = np.where(g_mean == 0, 1e-6, g_mean)
                    ratios = s_mean / g_mean
                else:
                    g_mean = np.where(grid_prices == 0, 1e-6, grid_prices)
                    ratios = selling_prices / g_mean
            else:
                # Handle different dimensions (should be rare)
                s_flat = selling_prices.flatten() if selling_prices.ndim > 1 else selling_prices
                g_flat = grid_prices.flatten() if grid_prices.ndim > 1 else grid_prices
                # Take only common length
                min_len = min(len(s_flat), len(g_flat))
                s_flat, g_flat = s_flat[:min_len], g_flat[:min_len]
                g_flat = np.where(g_flat == 0, 1e-6, g_flat)
                ratios = s_flat / g_flat
            
            data_by_mechanism[mechanism_type]['price_ratios'].append(ratios)
            
            # Store grid_fee value with price ratios for analysis
            if grid_fee is not None:
                # Calculate average price ratio (last 100 episodes)
                if len(ratios) >= 100:
                    avg_ratio = np.mean(ratios[-100:])
                    data_by_mechanism[mechanism_type]['grid_fees'].append((grid_fee, avg_ratio))
                    
    except Exception as e:
        print(f"Could not load price data for {run_dir}: {e}")


def _load_trading_profits(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load trading profit data for a specific run."""
    try:
        profit_file = os.path.join(run_dir, "data", "ddpg__trading_profit.pkl")
        
        if os.path.exists(profit_file):
            with open(profit_file, "rb") as f:
                trading_profits = pickle.load(f)
                
            # Convert to numpy if not already
            trading_profits = np.array(trading_profits)
            
            # Handle dimensionality
            if trading_profits.ndim > 1:
                # Mean across houses/agents
                trading_profits = np.mean(trading_profits, axis=1)
            
            # Calculate cumulative sum
            cum_profits = np.cumsum(trading_profits)
            data_by_mechanism[mechanism_type]['trading_profits'].append(cum_profits)
    except Exception as e:
        print(f"Could not load trading profit for {run_dir}: {e}")


def _load_p2p_energy_data(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load P2P energy trading data for a specific run."""
    try:
        p2p_file = os.path.join(run_dir, "data", "ddpg__energy_bought_p2p.pkl")
        
        if os.path.exists(p2p_file):
            with open(p2p_file, "rb") as f:
                p2p_energy = pickle.load(f)
                
            # Convert to numpy if not already
            p2p_energy = np.array(p2p_energy)
            
            # Handle dimensionality
            if p2p_energy.ndim > 1:
                # Mean across houses/agents
                p2p_energy = np.mean(p2p_energy, axis=1)
                
            data_by_mechanism[mechanism_type]['p2p_energy'].append(p2p_energy)
    except Exception as e:
        print(f"Could not load P2P energy for {run_dir}: {e}")


def _load_temperature_data(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load temperature penalty data for a specific run."""
    try:
        penalty_file = os.path.join(run_dir, "data", "ddpg__penalty.pkl")
        
        # Get comfort_penalty from hyperparameters
        comfort_penalty = None
        for hyper_data in data_by_mechanism[mechanism_type]['hyperparameters']:
            if hyper_data['run_id'] == run_id and 'params' in hyper_data:
                if 'environment' in hyper_data['params']:
                    if 'comfort_penalty' in hyper_data['params']['environment']:
                        comfort_penalty = hyper_data['params']['environment']['comfort_penalty']
                    elif 'temperature_comfort_penalty_weight' in hyper_data['params']['environment']:
                        comfort_penalty = hyper_data['params']['environment']['temperature_comfort_penalty_weight']
                    break
        
        if os.path.exists(penalty_file):
            with open(penalty_file, "rb") as f:
                penalty_data = pickle.load(f)
                
            # Convert to numpy if not already
            penalty_data = np.array(penalty_data)
            
            # Store penalty data (will be used as proxy for temperature control)
            if penalty_data.ndim > 1:
                # Mean across houses/agents
                penalty_data = np.mean(penalty_data, axis=1)
                
            data_by_mechanism[mechanism_type]['penalty'].append(penalty_data)
            
            # Store comfort_penalty value with temperature penalties for analysis
            if comfort_penalty is not None:
                # Calculate average penalty (last 100 episodes)
                if len(penalty_data) >= 100:
                    avg_penalty = np.mean(penalty_data[-100:])
                    data_by_mechanism[mechanism_type]['comfort_penalties'].append((comfort_penalty, avg_penalty))
            
            # Store for temperature visualization
            data_by_mechanism[mechanism_type]['temperatures'].append({
                'run_id': run_id,
                'penalties': penalty_data
            })
    except Exception as e:
        print(f"Could not load temperature data for {run_dir}: {e}")


def _load_battery_data(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load battery-related data for a specific run."""
    try:
        hvac_file = os.path.join(run_dir, "data", "ddpg__HVAC_energy_cons.pkl")
        depreciation_file = os.path.join(run_dir, "data", "ddpg__depreciation.pkl")
        
        battery_data = {}
        
        if os.path.exists(hvac_file):
            with open(hvac_file, "rb") as f:
                hvac_energy = pickle.load(f)
                if isinstance(hvac_energy, np.ndarray) and hvac_energy.ndim > 1:
                    hvac_energy = np.mean(hvac_energy, axis=1)
                data_by_mechanism[mechanism_type]['hvac_energy'].append(hvac_energy)
                battery_data['hvac_energy'] = hvac_energy
        
        if os.path.exists(depreciation_file):
            with open(depreciation_file, "rb") as f:
                depreciation = pickle.load(f)
                if isinstance(depreciation, np.ndarray) and depreciation.ndim > 1:
                    depreciation = np.mean(depreciation, axis=1)
                data_by_mechanism[mechanism_type]['depreciation'].append(depreciation)
                battery_data['depreciation'] = depreciation
        
        # Store for battery visualization
        if battery_data:
            battery_data['run_id'] = run_id
            data_by_mechanism[mechanism_type]['battery_data'].append(battery_data)
    except Exception as e:
        print(f"Could not load battery data for {run_dir}: {e}")


def _load_anticartel_penalties(run_dir, run_id, mechanism_type, data_by_mechanism):
    """Load anti-cartel penalties for a specific run."""
    try:
        # Path to the presumed anti-cartel penalty file
        penalties_file = os.path.join(run_dir, "data", "ddpg__anti_cartel_penalty.pkl") 
        # This filename needs to be correct for actual data loading.
        # If it's consistently not found, the plot will use synthetic data.

        if os.path.exists(penalties_file):
            with open(penalties_file, "rb") as f:
                penalty_data = pickle.load(f)
                
                # Original simple processing: convert to numpy array and ensure 1D
                penalty_data = np.array(penalty_data) # Ensure it's an array
                if penalty_data.ndim > 1:
                    # If multi-dimensional (e.g., per agent), average over the agents (axis 1)
                    # This assumes episodes are axis 0. Adjust if your data is structured differently.
                    penalty_data = np.mean(penalty_data, axis=1)
                
                # Final check to ensure it's a 1D array before appending
                if penalty_data.ndim == 1:
                    data_by_mechanism[mechanism_type]['anti_cartel_penalties'].append(penalty_data)
                else:
                    # This case should be rare if the above processing is correct
                    print(f"    WARNING: Processed anti-cartel penalty data for run {run_id} is not 1D (shape: {penalty_data.shape}). Skipping.")
        # else: # If file not found, do nothing, synthetic data will be used by plot func
            # print(f"    INFO: Anti-cartel penalty file not found at {penalties_file}. Plot will use synthetic data if no runs provide it.")

    except Exception as e:
        # Catch any other error during loading/processing of this specific file
        print(f"    ERROR: Could not load or process anti-cartel penalties for {run_dir} from {penalties_file}: {e}")