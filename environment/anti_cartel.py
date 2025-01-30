from collections import deque

import numpy as np

from hyperparameters import Config


class AntiCartelMechanism:
    def __init__(self):
        config = Config()
        
        # Load mechanism type and general parameters
        self.mechanism_type = config.get('anti_cartel', 'mechanism_type')  # 'detection', 'ceiling', or None
        
        # Only load other parameters if a mechanism is active
        if self.mechanism_type is not None:
            # Detection and Penalty System parameters
            self.monitoring_window = config.get('anti_cartel', 'monitoring_window') # Only used if type = "detection"
            self.similarity_threshold = config.get('anti_cartel', 'similarity_threshold') # Only used if type = "detection"
            self.penalty_factor = config.get('anti_cartel', 'penalty_factor') # Used by both mechanisms
            
            # Price Ceiling parameters
            self.markup_limit = config.get('anti_cartel', 'markup_limit') # Only used if type = "ceiling"
            self.market_elasticity = config.get('anti_cartel', 'market_elasticity') #   Only used if type = "ceiling"
            
            # Initialize price history for detection mechanism
            self.price_history = deque(maxlen=self.monitoring_window)
            self.current_episode_prices = []
            
            print(f"Anti-cartel mechanism initialized with type: {self.mechanism_type}")
        else:
            print("Anti-cartel mechanism disabled - operating in free market mode")
    
    def is_active(self):
        """Check if any anti-cartel mechanism is active"""
        return self.mechanism_type is not None
        
    def update_price_history(self, selling_prices, episode_done=False):
        """
        Update the price history with new selling prices
        Args:
            selling_prices: List of current selling prices for each house
            episode_done: Boolean indicating if the current episode is complete
        """
        # Only track prices if using detection mechanism
        if self.mechanism_type != 'detection':
            return
            
        self.current_episode_prices.append(selling_prices)
        
        if episode_done:
            # Calculate average prices for the episode
            if self.current_episode_prices:  # Check if we have any prices
                episode_avg_prices = np.mean(np.array(self.current_episode_prices), axis=0)
                self.price_history.append(episode_avg_prices)
                self.current_episode_prices = []

    def calculate_penalties(self, selling_prices, grid_price):
        """
        Calculate penalties for each house based on the selected mechanism
        Args:
            selling_prices: List of current selling prices for each house
            grid_price: Current grid price
        Returns:
            penalties: List of penalties for each house
        """
        num_houses = len(selling_prices)
        
        # If no mechanism is active, return zero penalties
        if not self.is_active():
            return [0.0] * num_houses
            
        if self.mechanism_type == 'detection':
            return self._calculate_detection_penalties(selling_prices)
        elif self.mechanism_type == 'ceiling':
            return self._calculate_ceiling_penalties(selling_prices, grid_price)
        
        # Fallback case (should never happen due to is_active() check)
        return [0.0] * num_houses
    
    def _calculate_detection_penalties(self, selling_prices):
        """
        Calculate penalties based on price pattern detection with robust handling of edge cases
        """
        if len(self.price_history) < 2:  # Need at least 2 episodes for comparison
            return [0.0] * len(selling_prices)
            
        penalties = [0.0] * len(selling_prices)
        price_history_array = np.array(list(self.price_history))
        
        # Calculate price correlations between houses
        for i in range(len(selling_prices)):
            for j in range(i + 1, len(selling_prices)):
                # Skip if either house has no price history
                if i >= price_history_array.shape[1] or j >= price_history_array.shape[1]:
                    continue
                    
                # Get price histories for both houses
                prices_i = price_history_array[:, i]
                prices_j = price_history_array[:, j]
                
                # Check for valid price variations
                std_i = np.std(prices_i)
                std_j = np.std(prices_j)
                
                # If prices are constant (zero std dev), they might be colluding
                if std_i < 1e-6 and std_j < 1e-6:
                    if abs(prices_i[0] - prices_j[0]) < 1e-6:  # Same constant prices
                        penalties[i] += self.penalty_factor * selling_prices[i]
                        penalties[j] += self.penalty_factor * selling_prices[j]
                    continue
                
                # Skip if either house has no price variation
                if std_i < 1e-6 or std_j < 1e-6:
                    continue
                    
                try:
                    # Calculate correlation safely
                    correlation = np.corrcoef(prices_i, prices_j)[0, 1]
                    
                    # Check if correlation is valid and high enough
                    if not np.isnan(correlation) and correlation > self.similarity_threshold:
                        penalties[i] += self.penalty_factor * selling_prices[i]
                        penalties[j] += self.penalty_factor * selling_prices[j]
                except Exception as e:
                    print(f"Warning: Error calculating correlation: {e}")
                    continue
                        
        return penalties
        
    def _calculate_ceiling_penalties(self, selling_prices, grid_price):
        """
        Calculate penalties based on oligopolistic price ceiling
        """
        penalties = [0.0] * len(selling_prices)
        price_ceiling = self.get_price_ceiling(grid_price)
        
        # Apply penalties for prices above ceiling
        for i, price in enumerate(selling_prices):
            if price > price_ceiling:
                penalties[i] = self.penalty_factor * (price - price_ceiling)
                
        return penalties

    def get_price_ceiling(self, grid_price):
        """
        Calculate and return the current price ceiling
        Args:
            grid_price: Current grid price from the utility
        Returns:
            price_ceiling: Maximum allowed selling price
        """
        if not self.is_active() or self.mechanism_type != 'ceiling':
            return float('inf')
            
        n = len(self.current_episode_prices[0]) if self.current_episode_prices else 1
        max_markup = 1 / (n * abs(self.market_elasticity))
        return grid_price * (1 + max_markup)

    def get_mechanism_status(self):
        """
        Get current status of the anti-cartel mechanism
        Returns:
            dict: Status information about the current mechanism
        """
        status = {
            'type': self.mechanism_type,
            'active': self.is_active()
        }
        
        if self.is_active():
            if self.mechanism_type == 'detection':
                status.update({
                    'monitoring_window': self.monitoring_window,
                    'episodes_monitored': len(self.price_history),
                    'similarity_threshold': self.similarity_threshold
                })
            elif self.mechanism_type == 'ceiling':
                status.update({
                    'markup_limit': self.markup_limit,
                    'market_elasticity': self.market_elasticity
                })
                
        return status