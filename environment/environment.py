import random
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import torch

from hyperparameters import Config
from utilities import Utilities
from .anti_cartel import AntiCartelMechanism


class Environment:
    """
    Smart Home Energy Management Environment.
    
    This environment simulates a group of houses where each can consume, generate, buy,
    and sell energy. Houses can trade energy with each other through a peer-to-peer
    market and interact with the utility grid.
    
    The environment exposes these key concepts:
    - House State: Inside temperature, ambient temperature, battery level, power generation, 
                  power demand, etc.
    - Actions:
        1. HVAC adjustment (e_t)
        2. Battery charge/discharge (a_batt)
        3. Setting a selling price (bounded by the grid price)
    
    Over each time step, the environment updates:
    - Energy balance, temperature, and battery usage based on house actions
    - Costs, trading profits, and penalties if temperatures exceed thresholds
    - Peer-to-peer energy transactions among houses with excess energy and those with deficits
    """
    
    def __init__(self, dynamic: bool = False, eval_mode: bool = False):
        """
        Initialize the environment with houses, market mechanics, and data sources.
        
        Args:
            dynamic: Whether to use dynamic data loading during episodes
            eval_mode: Whether to run in evaluation mode (fixed start point)
        """
        self.config = Config()
        self.eval = eval_mode
        self.dynamic = dynamic
        self.num_houses = self.config.get('environment', 'num_houses')
        self.utilities = Utilities(num_houses=self.num_houses)
        self.anti_cartel = AntiCartelMechanism()

        # Define state components with descriptive functions
        self._initialize_state_components()
        
        # Calculate and set dimensions
        self._initialize_dimensions()

        # Initialize house states (temperatures, batteries)
        self._initialize_house_states()

        # Load price parameters including ceiling and grid fees
        self._initialize_price_parameters()

        # Set random seed if specified in config
        self._set_random_seed()

        # Load required data for simulation
        self._initialize_data_loading()
        self.random_start_idx = self._select_random_start_idx()
        self._load_data()

        # Initialize time tracking
        self.time = 0
        self.done = False
    
    def _initialize_state_components(self) -> None:
        """Define state components for each house with accessor functions."""
        self.state_components = {
            "inside_temperature": lambda i: self.inside_temperatures[i],
            "ambient_temperature": lambda i: self.ambient_temperature,
            "sun_power": lambda i: self.sun_power[i],
            "price": lambda i: self.price,
            "battery": lambda i: self.batteries[i],
            "power_demand": lambda i: self.power_demand[i],
            "hour_of_day": lambda i: self.hour_of_day,
        }
    
    def _initialize_dimensions(self) -> None:
        """Calculate and store state and action dimensions."""
        # Base state dimension without other houses' selling prices
        self.BASE_STATE_DIM_PER_HOUSE = len(self.state_components)
        
        # Total state dimension including other houses' selling prices
        self.STATE_DIM_PER_HOUSE = self.BASE_STATE_DIM_PER_HOUSE + self.num_houses
        
        # Action dimensions (constant)
        self.ACTION_DIM_PER_HOUSE = 3  # e_t, a_batt, selling_price

        # Action bounds
        self.ACTION_BOUNDS = {
            'e_t': self.config.get('environment', 'hvac_action_bounds'),
            'a_batt': self.config.get('environment', 'battery_action_bounds'),
            'selling_price': [0.5, 0.95]  
        }

        # Calculate total dimensions
        self._state_dim = self.STATE_DIM_PER_HOUSE * self.num_houses
        self._action_dim = self.ACTION_DIM_PER_HOUSE * self.num_houses

        # Update config with calculated dimensions
        self.config.set('environment', 'state_dim_per_house', self.STATE_DIM_PER_HOUSE)
        self.config.set('environment', 'action_dim_per_house', self.ACTION_DIM_PER_HOUSE)
        self.config.set('environment', 'total_state_dim', self._state_dim)
        self.config.set('environment', 'total_action_dim', self._action_dim)
    
    def _initialize_house_states(self) -> None:
        """Initialize temperature, battery, and pricing states for all houses."""
        # Initialize selling prices
        self.selling_prices = [0.0 for _ in range(self.num_houses)]
        
        # Load parameters from config
        initial_temp = self.config.get('environment', 'initial_inside_temperature')
        battery_min = self.config.get('environment', 'battery_capacity_min')
        battery_max = self.config.get('environment', 'battery_capacity_max')
        
        # Initialize temperatures and batteries
        self.inside_temperatures = [initial_temp for _ in range(self.num_houses)]
        self.batteries = [
            round(random.uniform(battery_min + 1, battery_max), 1) 
            for _ in range(self.num_houses)
        ]
    
    def _initialize_price_parameters(self) -> None:
        """Initialize parameters related to pricing and market operations."""
        # Grid transaction fee
        self.grid_fee = self.config.get('environment', 'grid_fee')
        
        # Load price ceiling parameters
        price_ceiling_config = self.config.get('environment', 'price_ceiling')
        self.price_ceiling_enabled = price_ceiling_config.get('enabled', True)
        self.markup_factor = price_ceiling_config.get('markup_factor', 1.2)
        self.base_margin = price_ceiling_config.get('base_margin', 0.1)
    
    def _set_random_seed(self) -> None:
        """Set random seed for reproducibility if specified in config."""
        random_seed = self.config.get('simulation', 'random_seed')
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def _initialize_data_loading(self) -> None:
        """Initialize data sources and load paths."""
        self.weather_data_path = self.config.get('paths', 'weather_data')
        self.price_data_path = self.config.get('paths', 'price_data')

        # Load weather and price data
        self.ambient_temps_df = pd.read_csv(self.weather_data_path, header=3)
        self.price_df = pd.read_csv(self.price_data_path, header=0)

        # House assignment logic
        households = [3, 4, 6]  # Sample households used for data
        self._load_house_data(households)
    
    def _load_house_data(self, households: List[int]) -> None:
        """
        Load and process house-specific consumption and generation data.
        
        Args:
            households: List of household IDs to use for data sources
        """
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

            # Update total_hours based on minimum length of available data
            if total_hours is None or total_hours > len(cons_df):
                total_hours = len(cons_df)
            if total_hours > len(gen_df):
                total_hours = len(gen_df)

        # Assign data to houses (with repeating if necessary)
        self.agent_data = []
        for i in range(self.num_houses):
            idx = i % len(households)
            self.agent_data.append({
                'consumption_df': self.consumption_dfs[idx].iloc[:total_hours],
                'generation_df': self.generation_dfs[idx].iloc[:total_hours]
            })

        # Truncate weather and price data to match
        self.ambient_temps_df = self.ambient_temps_df.iloc[:total_hours]
        self.price_df = self.price_df.iloc[:total_hours]
        self.total_hours = total_hours
        
        # Set simulation parameters
        self.num_hours = self.config.get('simulation', 'num_hours')
        self.num_time_steps = self.num_hours
    
    def add_state_component(self, name: str, value_function: callable) -> None:
        """
        Add a new component to the state representation.
        
        Args:
            name: Name of the component
            value_function: Function that takes house index and returns component value
        """
        self.state_components[name] = value_function
        
        # Recalculate dimensions to include the new component
        self._recalculate_dimensions()
        
    def remove_state_component(self, name: str) -> None:
        """
        Remove a component from the state representation.
        
        Args:
            name: Name of the component to remove
        """
        if name in self.state_components:
            del self.state_components[name]
            
            # Recalculate dimensions to reflect the removal
            self._recalculate_dimensions()
    
    def _recalculate_dimensions(self) -> None:
        """Recalculate state dimensions after adding or removing components."""
        # Update base state dimension
        self.BASE_STATE_DIM_PER_HOUSE = len(self.state_components)
        
        # Update per-house state dimension (base + selling prices)
        self.STATE_DIM_PER_HOUSE = self.BASE_STATE_DIM_PER_HOUSE + self.num_houses
        
        # Calculate total state dimension
        self._state_dim = self.STATE_DIM_PER_HOUSE * self.num_houses
        
        # Update config with new dimensions
        self.config.set('environment', 'state_dim_per_house', self.STATE_DIM_PER_HOUSE)
        self.config.set('environment', 'total_state_dim', self._state_dim)
        
        print(f"State dimensions recalculated: {self.STATE_DIM_PER_HOUSE} per house, {self._state_dim} total")

    @property
    def state_dim(self) -> int:
        """Total state dimension for all houses."""
        return self.num_houses * self.STATE_DIM_PER_HOUSE

    @property
    def action_dim(self) -> int:
        """Total action dimension for all houses."""
        return self.num_houses * self.ACTION_DIM_PER_HOUSE

    @property
    def current_state(self) -> List[float]:
        """Current state of the environment."""
        return self._get_state()

    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get information about the action space.
        
        Returns:
            Dict containing action space dimensions, bounds, and house count
        """
        return {
            'dim_per_house': self.ACTION_DIM_PER_HOUSE,
            'total_dim': self.action_dim,
            'bounds': self.ACTION_BOUNDS,
            'num_houses': self.num_houses
        }

    def get_state_space_info(self) -> Dict[str, Any]:
        """
        Get information about the state space.
        
        Returns:
            Dict containing state space dimensions and house count
        """
        return {
            'dim_per_house': self.STATE_DIM_PER_HOUSE,
            'total_dim': self.state_dim,
            'num_houses': self.num_houses
        }

    def step(self, actions: torch.Tensor, A: float = 0.14 * 1000 / ((1 - 32) * 5 / 9)) -> Tuple[List[float], List[float], bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            actions: Tensor of shape [num_houses, 3] containing [e_t, a_batt, selling_price] for each house
            A: HVAC system thermal mass coefficient (default derived from physical constants)
            
        Returns:
            state: Current state after taking actions
            rewards: List of rewards for each house
            done: Boolean indicating if episode is finished
            infos: Dictionary containing additional information about the step
        """
        # Convert actions from normalized space to actual values
        actions = self.utilities.unscaler(actions)
        
        # Initialize tracking dictionaries for various metrics
        infos = self._initialize_info_tracking()
        rewards = []  # Will store final rewards for each house

        # Set selling prices with anti-cartel mechanism
        self._set_selling_prices(actions, infos)
        
        # Calculate energy balance for each house
        house_energy_status = self._calculate_energy_balances(actions)
        
        # Execute P2P energy trading
        transactions = self._execute_p2p_trading(house_energy_status)
        
        # Update houses and calculate rewards
        rewards = self._update_houses_and_calculate_rewards(actions, house_energy_status, transactions, infos, A)
        
        # Apply anti-cartel mechanism if active
        rewards = self._apply_anti_cartel_mechanism(rewards, infos)
        
        # Update time and environment state
        self.time += 1
        self.done = self.time >= self.num_time_steps
        
        if not self.done and self.dynamic:
            self._update_dynamic_variables()
        
        return self._get_state(), rewards, self.done, infos
    
    def _initialize_info_tracking(self) -> Dict[str, Any]:
        """
        Initialize tracking dictionary for step metrics.
        
        Returns:
            Empty info dictionary with required tracking keys
        """
        return {
            'HVAC_energy_cons': [],      # Energy consumption from HVAC
            'depreciation': [],          # Battery depreciation costs
            'penalty': [],               # Temperature violation penalties
            'trading_profit': [],        # Profits from P2P energy trading
            'energy_bought_p2p': [],     # Amount of energy bought P2P
            'selling_prices': [],        # Final selling prices
            'anti_cartel_penalties': [], # Penalties from anti-cartel mechanism
            'grid_prices': float(self.price)  # Current grid price
        }
    
    def _set_selling_prices(self, actions: torch.Tensor, infos: Dict[str, Any]) -> None:
        """
        Set selling prices for each house based on actions and constraints.
        
        Args:
            actions: Processed actions from the agent
            infos: Info dictionary to update with price information
        """
        current_grid_price = self.price
        self.selling_prices = []
        
        # Get price ceiling if using ceiling mechanism
        if self.anti_cartel.mechanism_type == 'ceiling':
            price_ceiling = self.anti_cartel.get_price_ceiling(current_grid_price)
        else:
            price_ceiling = float('inf')
        
        # Set selling prices for each house with appropriate bounds
        for i in range(self.num_houses):
            # Convert normalized action to actual price
            raw_price = actions[i, 2].item() * current_grid_price
            
            if self.anti_cartel.mechanism_type == 'ceiling':
                # For ceiling mechanism: enforce price must be below ceiling
                bounded_price = min(max(0, raw_price), price_ceiling)
            else:
                # For other mechanisms: allow prices up to grid price
                bounded_price = min(max(0, raw_price), current_grid_price)
            
            self.selling_prices.append(bounded_price)
    
    def _calculate_energy_metrics(self, house_idx: int, house_energy_status: List[Dict[str, Any]], transactions: Dict[int, List[Dict[str, Any]]], actions: torch.Tensor) -> Dict[str, float]:
        """
        Calculate energy-related metrics for a specific house.
        
        Args:
            house_idx: Index of the house
            house_energy_status: Energy status for each house
            transactions: P2P energy transactions
            actions: Processed actions
            
        Returns:
            Dict of energy-related metrics for the house
        """
        # Get house energy status
        house_status = next(h for h in house_energy_status if h['house_id'] == house_idx)
        
        # Calculate energy consumed and sources
        total_consumption = house_status['deficit'] if house_status['deficit'] > 0 else 0
        energy_from_p2p = sum(t['amount'] for t in transactions[house_idx])
        energy_from_grid = max(0, total_consumption - energy_from_p2p)
        
        # Calculate trading profits as seller
        trading_profit = 0
        for buyer_id, buyer_transactions in transactions.items():
            for t in buyer_transactions:
                if t['seller_id'] == house_idx:
                    trading_profit += t['cost'] / (1 + self.grid_fee)
        
        # Calculate costs
        hvac_energy_cost = (
            1e-3 * self.price * energy_from_grid +
            sum(t['cost'] for t in transactions[house_idx])
        )
        
        # Battery depreciation cost
        a_batt = actions[house_idx, 1].item()
        battery_depreciation = self.config.get('cost_model', 'depreciation_coeff') * abs(a_batt)
        
        return {
            'hvac_energy_cost': hvac_energy_cost,
            'battery_depreciation': battery_depreciation,
            'trading_profit': trading_profit,
            'energy_from_p2p': energy_from_p2p
        }
    
    def _calculate_temperature_penalty(self, house_idx: int) -> float:
        """
        Calculate temperature comfort penalty for a house.
        
        Args:
            house_idx: Index of the house
            
        Returns:
            Temperature penalty value (0 if within comfort range)
        """
        temp = self.inside_temperatures[house_idx]
        temp_max = self.config.get('environment', 'temperature_max')
        temp_min = self.config.get('environment', 'temperature_min')
        
        if temp >= temp_max:
            return temp - temp_max
        elif temp <= temp_min:
            return temp_min - temp
        else:
            return 0
    
    def _calculate_reward(self, energy_metrics: Dict[str, float], temp_penalty: float) -> float:
        """
        Calculate the final reward for a house based on energy metrics and temperature penalty.
        
        Args:
            energy_metrics: Dict containing energy-related costs and profits
            temp_penalty: Temperature comfort penalty value
            
        Returns:
            Total reward value
        """
        beta = self.config.get('reward', 'beta')
        
        # Calculate total cost (negative component)
        total_cost = (
            energy_metrics['hvac_energy_cost'] + 
            energy_metrics['battery_depreciation'] + 
            temp_penalty
        )
        
        # Final reward is negative costs plus trading profit
        reward = -beta * total_cost + energy_metrics['trading_profit']
        
        return reward
    
    def _apply_anti_cartel_mechanism(self, rewards: List[float], infos: Dict[str, Any]) -> List[float]:
        """
        Apply anti-cartel mechanism penalties if active.
        
        Args:
            rewards: List of current rewards for each house
            infos: Info dictionary to update with penalties
            
        Returns:
            Updated list of rewards after applying penalties
        """
        if self.anti_cartel.is_active():
            penalties = self.anti_cartel.calculate_penalties(self.selling_prices, self.price)
            rewards = [r - p for r, p in zip(rewards, penalties)]
            infos['anti_cartel_penalties'] = penalties
            
            if self.anti_cartel.mechanism_type == 'detection':
                self.anti_cartel.update_price_history(self.selling_prices)
        
    def _calculate_energy_balances(self, actions: torch.Tensor) -> List[Dict[str, Any]]:

        """
        Calculate energy surplus or deficit for each house based on consumption and generation.
        
        Args:
            actions: Processed actions from the agent
            
        Returns:
            List of dicts with energy status for each house
        """
        house_energy_status = []
        for i in range(self.num_houses):
            # Get HVAC energy consumption from action
            e_t = actions[i, 0].item()
            # Calculate total consumption including base load
            total_consumption = self.power_demand[i] + 1e-3 * e_t
            # Get available solar power
            available_power = self.sun_power[i]
            
            # Calculate energy surplus or deficit
            excess = max(0, available_power - total_consumption)
            deficit = max(0, total_consumption - available_power)
            
            house_energy_status.append({
                'house_id': i,
                'excess': excess,
                'deficit': deficit,
                'selling_price': self.selling_prices[i]
            })
        
        return house_energy_status
    
    def _execute_p2p_trading(self, house_energy_status: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Execute P2P energy trading between houses with excess energy and those with deficits.
        
        Args:
            house_energy_status: Energy status for each house
            
        Returns:
            Dict mapping buyer house IDs to their transactions
        """
        # Sort houses into buyers and sellers
        sellers = sorted(
            [h for h in house_energy_status if h['excess'] > 0],
            key=lambda x: x['selling_price']  # Sort by price ascending
        )
        buyers = [h for h in house_energy_status if h['deficit'] > 0]
        
        # Track all transactions
        transactions = {i: [] for i in range(self.num_houses)}
        
        # Match buyers with sellers in order of price (lowest first)
        for buyer in buyers:
            remaining_demand = buyer['deficit']
            buyer_id = buyer['house_id']
            
            for seller in sellers:
                if remaining_demand <= 0 or seller['excess'] <= 0:
                    continue
                    
                seller_id = seller['house_id']
                selling_price = seller['selling_price']
                
                # Only trade if selling price is beneficial compared to grid
                if selling_price < self.price:
                    # Calculate energy and cost for this transaction
                    energy_traded = min(remaining_demand, seller['excess'])
                    transaction_cost = energy_traded * selling_price * (1 + self.grid_fee)
                    
                    # Record transaction
                    transactions[buyer_id].append({
                        'seller_id': seller_id,
                        'amount': energy_traded,
                        'cost': transaction_cost
                    })
                    
                    # Update remaining quantities
                    remaining_demand -= energy_traded
                    seller['excess'] -= energy_traded
        
        return transactions
    
    def _update_houses_and_calculate_rewards(
        self, 
        actions: torch.Tensor,
        house_energy_status: List[Dict[str, Any]],
        transactions: Dict[int, List[Dict[str, Any]]],
        infos: Dict[str, Any],
        A: float
    ) -> List[float]:
        """
        Update house states and calculate rewards based on actions and transactions.
        
        Args:
            actions: Processed actions
            house_energy_status: Energy status for each house
            transactions: P2P energy transactions
            infos: Info dictionary to update
            A: HVAC system thermal mass coefficient
            
        Returns:
            List of rewards for each house
        """
        rewards = []
        
        for i in range(self.num_houses):
            # Get actions for this house
            e_t = actions[i, 0].item()
            a_batt = actions[i, 1].item()
            
            # Update temperature state
            self._update_temperature(i, e_t, A)
            
            # Update battery state
            self._update_battery_state(i, a_batt)
            
            # Calculate energy costs and trading profits
            energy_metrics = self._calculate_energy_metrics(i, house_energy_status, transactions, actions)
            
            # Calculate temperature penalty
            temp_penalty = self._calculate_temperature_penalty(i)
            
            # Calculate final reward for this house
            reward = self._calculate_reward(energy_metrics, temp_penalty)
            
            # Store all metrics
            rewards.append(reward)
            infos['HVAC_energy_cons'].append(energy_metrics['hvac_energy_cost'])
            infos['depreciation'].append(energy_metrics['battery_depreciation'])
            infos['penalty'].append(temp_penalty)
            infos['trading_profit'].append(energy_metrics['trading_profit'])
            infos['energy_bought_p2p'].append(energy_metrics['energy_from_p2p'])
            infos['selling_prices'].append(self.selling_prices[i])
        
        return rewards
    
    def _update_temperature(self, house_idx: int, e_t: float, A: float) -> None:
        """
        Update temperature state for a house based on HVAC action.
        
        Args:
            house_idx: Index of the house
            e_t: HVAC energy consumption action
            A: HVAC system thermal mass coefficient
        """
        weight = self.config.get('environment', 'temperature_comfort_penalty_weight')
        efficiency = self.config.get('environment', 'hvac_efficiency')
        
        self.inside_temperatures[house_idx] = weight * self.inside_temperatures[house_idx] + \
            (1 - weight) * (self.ambient_temperature - (efficiency / A) * e_t)
    
    def _update_battery_state(self, house_idx: int, a_batt: float) -> None:
        """
        Update battery state for a house based on charge/discharge action.
        
        Args:
            house_idx: Index of the house
            a_batt: Battery action (positive for charging, negative for discharging)
        """
        if a_batt > 0:
            # Charging with efficiency loss
            self.batteries[house_idx] = min(
                self.batteries[house_idx] + self.config.get('environment', 'battery_charging_efficiency') * a_batt, 
                self.config.get('environment', 'battery_capacity_max')
            )
        else:
            # Discharging with efficiency loss
            self.batteries[house_idx] = max(
                self.batteries[house_idx] + a_batt / self.config.get('environment', 'battery_discharging_efficiency'), 
                self.config.get('environment', 'battery_capacity_min')
            )
    
    def _calculate_energy_metrics(self, house_idx: int, house_energy_status: List[Dict[str, Any]], transactions: Dict[int, List[Dict[str, Any]]], actions: torch.Tensor) -> Dict[str, float]:
        """
        Calculate energy-related metrics for a specific house.
        
        Args:
            house_idx: Index of the house
            house_energy_status: Energy status for each house
            transactions: P2P energy transactions
            actions: Processed actions
            
        Returns:
            Dict of energy-related metrics for the house
        """
        # Get house energy status
        house_status = next(h for h in house_energy_status if h['house_id'] == house_idx)
        
        # Calculate energy consumed and sources
        total_consumption = house_status['deficit'] if house_status['deficit'] > 0 else 0
        energy_from_p2p = sum(t['amount'] for t in transactions[house_idx])
        energy_from_grid = max(0, total_consumption - energy_from_p2p)
        
        # Calculate trading profits as seller
        trading_profit = 0
        for buyer_id, buyer_transactions in transactions.items():
            for t in buyer_transactions:
                if t['seller_id'] == house_idx:
                    trading_profit += t['cost'] / (1 + self.grid_fee)
        
        # Calculate costs
        hvac_energy_cost = (
            1e-3 * self.price * energy_from_grid +
            sum(t['cost'] for t in transactions[house_idx])
        )
        
        # Battery depreciation cost
        a_batt = actions[house_idx, 1].item()
        battery_depreciation = self.config.get('cost_model', 'depreciation_coeff') * abs(a_batt)
        
        return {
            'hvac_energy_cost': hvac_energy_cost,
            'battery_depreciation': battery_depreciation,
            'trading_profit': trading_profit,
            'energy_from_p2p': energy_from_p2p
        }
    
    def _calculate_temperature_penalty(self, house_idx: int) -> float:
        """
        Calculate temperature comfort penalty for a house.
        
        Args:
            house_idx: Index of the house
            
        Returns:
            Temperature penalty value (0 if within comfort range)
        """
        temp = self.inside_temperatures[house_idx]
        temp_max = self.config.get('environment', 'temperature_max')
        temp_min = self.config.get('environment', 'temperature_min')
        
        if temp >= temp_max:
            return temp - temp_max
        elif temp <= temp_min:
            return temp_min - temp
        else:
            return 0
    
    def _calculate_reward(self, energy_metrics: Dict[str, float], temp_penalty: float) -> float:
        """
        Calculate the final reward for a house based on energy metrics and temperature penalty.
        
        Args:
            energy_metrics: Dict containing energy-related costs and profits
            temp_penalty: Temperature comfort penalty value
            
        Returns:
            Total reward value
        """
        beta = self.config.get('reward', 'beta')
        
        # Calculate total cost (negative component)
        total_cost = (
            energy_metrics['hvac_energy_cost'] + 
            energy_metrics['battery_depreciation'] + 
            temp_penalty
        )
        
        # Final reward is negative costs plus trading profit
        reward = -beta * total_cost + energy_metrics['trading_profit']
        
        return reward
    
    def _apply_anti_cartel_mechanism(self, rewards: List[float], infos: Dict[str, Any]) -> List[float]:
        """
        Apply anti-cartel mechanism penalties if active.
        
        Args:
            rewards: List of current rewards for each house
            infos: Info dictionary to update with penalties
            
        Returns:
            Updated list of rewards after applying penalties
        """
        if self.anti_cartel.is_active():
            penalties = self.anti_cartel.calculate_penalties(self.selling_prices, self.price)
            rewards = [r - p for r, p in zip(rewards, penalties)]
            infos['anti_cartel_penalties'] = penalties
            
            if self.anti_cartel.mechanism_type == 'detection':
                self.anti_cartel.update_price_history(self.selling_prices)
        
        return rewards

    def _update_dynamic_variables(self) -> None:
        """Update dynamic variables for the next time step."""
        idx = self.time - 1
        self.ambient_temperature = self.ambient_temperatures[idx]
        self.price = self.prices[idx]
        self.sun_power = self.sun_powers[idx, :]
        self.sun_power_pred = self.sun_power_predictions[idx, :]
        self.power_demand = self.power_demands[idx, :]
        self.power_demand_pred = self.power_demand_predictions[idx, :]
        self.hour_of_day = (self.random_start_idx + idx) % 24

    def reset(self) -> List[float]:
        """
        Reset the environment to initial state and handle episode completion.
        
        Returns:
            Initial state of the environment
        """
        # If anti-cartel is active, mark the episode completion before reset
        if hasattr(self, 'anti_cartel') and self.anti_cartel.is_active():
            self.anti_cartel.update_price_history(self.selling_prices, episode_done=True)

        # Reset house states
        self._reset_house_states()
        
        # Reset time tracking
        self.time = 0
        self.done = False

        # Get new starting point and load data
        self.random_start_idx = self._select_random_start_idx()
        self._load_data()

        return self._get_state()
    
    def _reset_house_states(self) -> None:
        """Reset house state variables to initial values."""
        # Reset temperatures
        initial_temp = self.config.get('environment', 'initial_inside_temperature')
        self.inside_temperatures = [initial_temp for _ in range(self.num_houses)]
        
        # Reset batteries with random levels
        battery_min = self.config.get('environment', 'battery_capacity_min')
        battery_max = self.config.get('environment', 'battery_capacity_max')
        self.batteries = [
            round(random.uniform(battery_min + 1, battery_max), 1)
            for _ in range(self.num_houses)
        ]
        
        # Reset selling prices
        self.selling_prices = [0.0 for _ in range(self.num_houses)]

    def _select_random_start_idx(self) -> int:
        """
        Select random start index for episode.
        
        Returns:
            Starting index for data retrieval
        """
        if self.eval:
            return 0
        max_start_idx = self.total_hours - (self.num_hours + 1)
        return random.randint(0, max_start_idx)

    def _load_data(self) -> None:
        """Load data for the current episode from the selected starting index."""
        idx_start = self.random_start_idx
        idx_end = idx_start + self.num_time_steps + 1

        # Load temperature data
        self.ambient_temperatures = self.ambient_temps_df.iloc[idx_start:idx_end, 2].values
        self.ambient_temperature = self.ambient_temperatures[0]

        # Load price data
        self.prices = self.price_df.iloc[idx_start:idx_end, 0].values
        self.price = self.prices[0]

        # Initialize arrays for house-specific data
        self._load_house_specific_data(idx_start, idx_end)

        # Set initial values
        self.sun_power = self.sun_powers[0, :]
        self.sun_power_pred = self.sun_power_predictions[0, :]
        self.power_demand = self.power_demands[0, :]
        self.power_demand_pred = self.power_demand_predictions[0, :]
        self.hour_of_day = (self.random_start_idx) % 24
    
    def _load_house_specific_data(self, idx_start: int, idx_end: int) -> None:
        """
        Load house-specific generation and consumption data for all houses.
        
        Args:
            idx_start: Starting index for data
            idx_end: Ending index for data
        """
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

        # Convert to numpy arrays with shape (time_steps, num_houses)
        self.sun_powers = np.array(self.sun_powers).T
        self.sun_power_predictions = np.array(self.sun_power_predictions).T
        self.power_demands = np.array(self.power_demands).T
        self.power_demand_predictions = np.array(self.power_demand_predictions).T
    
    def _get_state(self) -> List[float]:
        """
        Get the current state of the environment for all houses.
        
        Returns:
            Flattened list of state values for all houses
        """
        state = []
        for i in range(self.num_houses):
            # Gather all state components dynamically using component accessor functions
            house_state = [component_func(i) for component_func in self.state_components.values()]
            
            # Add other houses' selling prices to the state
            house_state.extend([self.selling_prices[j] for j in range(self.num_houses)])
            
            # Add this house's state to the overall state
            state.extend(house_state)
            
        return state

    def get_anti_cartel_status(self) -> Dict[str, Any]:
        """
        Get the current status of the anti-cartel mechanism.
        
        Returns:
            Dictionary with status information about the anti-cartel mechanism
        """
        return self.anti_cartel.get_mechanism_status()