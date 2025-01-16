import random
import pandas as pd
import numpy as np
import torch

from hyperparameters import Config
from utilities import Utilities

class Environment:   
    def __init__(self, dynamic=False, eval_mode=False):
        config = Config()
        self.eval = eval_mode
        self.dynamic = dynamic
        self.num_houses = config.get('environment', 'num_houses')
        self.utilities = Utilities(num_houses=self.num_houses)

        # Load hyperparameters
        self.initial_inside_temperature = config.get('environment', 'initial_inside_temperature')
        self.battery_capacity_min = config.get('environment', 'battery_capacity_min')
        self.battery_capacity_max = config.get('environment', 'battery_capacity_max')
        self.num_hours = config.get('simulation', 'num_hours')
        self.random_seed = config.get('simulation', 'random_seed')
        self.epsilon = config.get('environment', 'epsilon')
        self.eta_hvac = config.get('environment', 'eta_hvac')
        self.n_c = config.get('environment', 'n_c')
        self.n_d = config.get('environment', 'n_d')
        self.depreciation_coeff = config.get('cost_model', 'depreciation_coeff')
        self.t_max = config.get('environment', 't_max')
        self.t_min = config.get('environment', 't_min')
        self.beta = config.get('reward', 'beta')
        self.num_time_steps = self.num_hours
        # Initialize selling prices for each house
        self.selling_prices = [0.0 for _ in range(self.num_houses)]
        
        # Grid transaction fee (5%)
        self.grid_fee = config.get('environment', 'grid_fee')

        self.ACTION_DIM = config.get('environment', 'action_dim_per_house')
        self.STATE_DIM_PER_HOUSE = config.get('environment', 'state_dim_per_house') + self.num_houses
        self.ACTION_BOUNDS = {
            'e_t': config.get('environment', 'hvac_action_bounds'),
            'a_batt': config.get('environment', 'battery_action_bounds'),
            'selling_price': [0, 1]  # Normalized selling price, will be scaled to grid price
        }
        # Initialize temperatures and batteries
        self.inside_temperatures = [self.initial_inside_temperature for _ in range(self.num_houses)]
        self.batteries = [
            round(random.uniform(self.battery_capacity_min + 1, self.battery_capacity_max), 1) 
            for _ in range(self.num_houses)
        ]

        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Load data
        self._initialize_data_loading(config)
        self.random_start_idx = self._select_random_start_idx()
        self._load_data()

        # Initialize time and done flag
        self.time = 0
        self.done = False

    @property
    def state_dim(self):
        """Total state dimension for all houses"""
        return self.num_houses * self.STATE_DIM_PER_HOUSE

    @property
    def action_dim(self):
        """Total action dimension for all houses"""
        return self.num_houses * self.ACTION_DIM

    @property
    def current_state(self):
        """Current state of the environment"""
        return self._get_state()

    def get_action_space_info(self):
        """Returns information about the action space"""
        return {
            'dim_per_house': self.ACTION_DIM,
            'total_dim': self.action_dim,
            'bounds': self.ACTION_BOUNDS,
            'num_houses': self.num_houses
        }

    def get_state_space_info(self):
        """Returns information about the state space"""
        return {
            'dim_per_house': self.STATE_DIM_PER_HOUSE,
            'total_dim': self.state_dim,
            'num_houses': self.num_houses
        }

    def _initialize_data_loading(self, config):
        """Initialize data loading parameters and paths"""
        self.weather_data_path = config.get('paths', 'weather_data')
        self.price_data_path = config.get('paths', 'price_data')

        # Load weather and price data
        self.ambient_temps_df = pd.read_csv(self.weather_data_path, header=3)
        self.price_df = pd.read_csv(self.price_data_path, header=0)

        # House assignment logic
        households = [3, 4, 6]
        self._load_house_data(households)

    def _load_house_data(self, households):
        """Load and process house-specific data"""
        self.consumption_dfs = []
        self.generation_dfs = []
        total_hours = None

        for house_num in households:
            # Load consumption prediction data
            cons_file = f'data/lstms_predictions/Consumption_prediction_house_{house_num}.csv'
            cons_df = pd.read_csv(cons_file)
            self.consumption_dfs.append(cons_df)

            # Load generation prediction data
            gen_file = f'data/lstms_predictions/Generation_prediction_house_{house_num}.csv'
            gen_df = pd.read_csv(gen_file)
            self.generation_dfs.append(gen_df)

            # Update total_hours based on minimum length
            if total_hours is None or total_hours > len(cons_df):
                total_hours = len(cons_df)
            if total_hours > len(gen_df):
                total_hours = len(gen_df)

        # Assign data to houses
        self.agent_data = []
        for i in range(self.num_houses):
            idx = i % len(households)
            self.agent_data.append({
                'consumption_df': self.consumption_dfs[idx].iloc[:total_hours],
                'generation_df': self.generation_dfs[idx].iloc[:total_hours]
            })

        # Truncate weather and price data
        self.ambient_temps_df = self.ambient_temps_df.iloc[:total_hours]
        self.price_df = self.price_df.iloc[:total_hours]
        self.total_hours = total_hours

    def step(self, actions, A=0.14 * 1000 / ((1 - 32) * 5 / 9)):
        """
        Advances the environment by one time step based on the actions for each house.
        Args:
            actions: Tensor of shape [num_houses, 3] containing [e_t, a_batt, selling_price] for each house
        Returns:
            state: Current state after taking the actions
            rewards: List of rewards for each house
            done: Boolean indicating if episode is finished
            infos: Dictionary containing additional information about the step
        """
        # Unscale actions
        actions = self.utilities.unscaler(actions)
        
        # Extract and update selling prices (normalized relative to grid price)
        for i in range(self.num_houses):
            self.selling_prices[i] = min(actions[i, 2].item() * self.price, self.price)
        
        rewards = []
        infos = {
            'HVAC_energy_cons': [], 
            'depreciation': [], 
            'penalty': [],
            'trading_profit': [],
            'energy_bought_p2p': [],
            'selling_prices': []
        }
        
        # Calculate initial energy balance for each house
        house_energy_status = []
        for i in range(self.num_houses):
            e_t = actions[i, 0].item()
            total_consumption = self.power_demand[i] + 1e-3 * e_t
            available_power = self.sun_power[i]
            
            excess = max(0, available_power - total_consumption)
            deficit = max(0, total_consumption - available_power)
            
            house_energy_status.append({
                'house_id': i,
                'excess': excess,
                'deficit': deficit,
                'selling_price': self.selling_prices[i],
            })
        
        # Separate houses into buyers and sellers
        sellers = [h for h in house_energy_status if h['excess'] > 0]
        buyers = [h for h in house_energy_status if h['deficit'] > 0]
        
        # Sort sellers by price for efficient matching
        sellers.sort(key=lambda x: x['selling_price'])
        
        # Initialize transaction records
        transactions = {i: [] for i in range(self.num_houses)}
        
        # Match buyers with sellers
        for buyer in buyers:
            remaining_demand = buyer['deficit']
            buyer_id = buyer['house_id']
            
            for seller in sellers:
                if remaining_demand <= 0 or seller['excess'] <= 0:
                    continue
                    
                seller_id = seller['house_id']
                selling_price = seller['selling_price']
                
                # Only buy if cheaper than grid price
                if selling_price < self.price:
                    # Calculate transaction amount
                    energy_traded = min(remaining_demand, seller['excess'])
                    transaction_cost = energy_traded * selling_price * (1 + self.grid_fee)
                    
                    # Record transaction
                    transactions[buyer_id].append({
                        'seller_id': seller_id,
                        'amount': energy_traded,
                        'cost': transaction_cost
                    })
                    
                    # Update remaining values
                    remaining_demand -= energy_traded
                    seller['excess'] -= energy_traded
        
        # Process updates and calculate rewards for each house
        for i in range(self.num_houses):
            e_t = actions[i, 0].item()
            a_batt = actions[i, 1].item()
            
            # Update inside temperature
            self.inside_temperatures[i] = self.epsilon * self.inside_temperatures[i] + \
                (1 - self.epsilon) * (self.ambient_temperature - (self.eta_hvac / A) * e_t)
            
            # Update battery state
            if a_batt > 0:
                self.batteries[i] = min(self.batteries[i] + self.n_c * a_batt, self.battery_capacity_max)
            else:
                self.batteries[i] = max(self.batteries[i] + a_batt / self.n_d, self.battery_capacity_min)
            
            # Calculate energy balance and transactions
            house_status = next(h for h in house_energy_status if h['house_id'] == i)
            total_consumption = house_status['deficit'] if house_status['deficit'] > 0 else 0
            
            # Calculate energy bought from peers
            energy_from_p2p = sum(t['amount'] for t in transactions[i])
            # Remaining energy needed from grid
            energy_from_grid = max(0, total_consumption - energy_from_p2p)
            
            # Calculate trading profits (from selling)
            trading_profit = 0
            for buyer_id, buyer_transactions in transactions.items():
                for t in buyer_transactions:
                    if t['seller_id'] == i:
                        # Seller gets price minus grid fee
                        trading_profit += t['cost'] / (1 + self.grid_fee)
            
            # Calculate costs
            hvac_energy_cons = (
                self.price * energy_from_grid +  # Cost of energy from grid
                sum(t['cost'] for t in transactions[i])  # Cost of energy from peers
            )
            depreciation = self.depreciation_coeff * abs(a_batt)
            
            # Calculate temperature penalty
            if self.inside_temperatures[i] >= self.t_max:
                temp_penalty = self.inside_temperatures[i] - self.t_max
            elif self.inside_temperatures[i] <= self.t_min:
                temp_penalty = self.t_min - self.inside_temperatures[i]
            else:
                temp_penalty = 0
            
            # Calculate final reward
            reward = -self.beta * (hvac_energy_cons + depreciation) + trading_profit - temp_penalty
            rewards.append(reward)
            
            # Store info
            infos['HVAC_energy_cons'].append(hvac_energy_cons)
            infos['depreciation'].append(depreciation)
            infos['penalty'].append(temp_penalty)
            infos['trading_profit'].append(trading_profit)
            infos['energy_bought_p2p'].append(energy_from_p2p)
            infos['selling_prices'].append(self.selling_prices[i]) 
            infos['grid_prices'] = float(self.price)
        
        # Update time and check if done
        self.time += 1
        self.done = self.time >= self.num_time_steps
        
        # Update environment state if not done
        if not self.done and self.dynamic:
            self._update_dynamic_variables()
        
        return self._get_state(), rewards, self.done, infos

    def _update_dynamic_variables(self):
        """Update dynamic variables for the next time step"""
        idx = self.time - 1
        self.ambient_temperature = self.ambient_temperatures[idx]
        self.price = self.prices[idx]
        self.sun_power = self.sun_powers[idx, :]
        self.sun_power_pred = self.sun_power_predictions[idx, :]
        self.power_demand = self.power_demands[idx, :]
        self.power_demand_pred = self.power_demand_predictions[idx, :]
        self.hour_of_day = (self.random_start_idx + idx) % 24

    def reset(self):
        """Resets the environment to initial state"""
        self.inside_temperatures = [self.initial_inside_temperature for _ in range(self.num_houses)]
        self.batteries = [
            round(random.uniform(self.battery_capacity_min + 1, self.battery_capacity_max), 1)
            for _ in range(self.num_houses)
        ]
        self.time = 0
        self.done = False

        self.random_start_idx = self._select_random_start_idx()
        self._load_data()

        return self._get_state()

    def _select_random_start_idx(self):
        """Select random start index for episode"""
        if self.eval:
            return 0
        max_start_idx = self.total_hours - (self.num_hours + 1)
        return random.randint(0, max_start_idx)

    def _load_data(self):
        """Load data for the current episode"""
        idx_start = self.random_start_idx
        idx_end = idx_start + self.num_time_steps + 1

        self.ambient_temperatures = self.ambient_temps_df.iloc[idx_start:idx_end, 2].values
        self.ambient_temperature = self.ambient_temperatures[0]

        self.prices = self.price_df.iloc[idx_start:idx_end, 0].values
        self.price = self.prices[0]

        self.sun_powers = []
        self.sun_power_predictions = []
        self.power_demands = []
        self.power_demand_predictions = []

        for data in self.agent_data:
            gen_df = data['generation_df']
            cons_df = data['consumption_df']

            self.sun_powers.append(gen_df['Actual Generation'].values[idx_start:idx_end-1])
            self.sun_power_predictions.append(gen_df['Predicted Generation'].values[idx_start+1:idx_end])
            self.power_demands.append(cons_df['Actual Consumption'].values[idx_start:idx_end-1])
            self.power_demand_predictions.append(cons_df['Predicted Consumption'].values[idx_start+1:idx_end])

        self.sun_powers = np.array(self.sun_powers).T
        self.sun_power_predictions = np.array(self.sun_power_predictions).T
        self.power_demands = np.array(self.power_demands).T
        self.power_demand_predictions = np.array(self.power_demand_predictions).T

        self.sun_power = self.sun_powers[0, :]
        self.sun_power_pred = self.sun_power_predictions[0, :]
        self.power_demand = self.power_demands[0, :]
        self.power_demand_pred = self.power_demand_predictions[0, :]

        self.hour_of_day = (self.random_start_idx) % 24
    
    def _get_state(self):
        """Returns the current state of the environment for all houses."""
        state = []
        for i in range(self.num_houses):
            house_state = [
                self.inside_temperatures[i],
                self.ambient_temperature,
                self.sun_power[i],
                self.sun_power_pred[i],
                self.price,
                self.batteries[i],
                self.power_demand[i],
                self.power_demand_pred[i],
                self.hour_of_day
            ]
            # Add other houses' selling prices to the state
            house_state.extend([self.selling_prices[j] for j in range(self.num_houses)])
            state.extend(house_state)
        return state