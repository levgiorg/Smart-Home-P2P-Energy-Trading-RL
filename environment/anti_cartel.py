from collections import deque

import numpy as np

from hyperparameters import Config


class AntiCartelMechanism:
    def __init__(self):
        config = Config()
        
        # Load mechanism type and general parameters
        self.mechanism_type = config.get('anti_cartel', 'mechanism_type')
        
        if self.mechanism_type is not None:
            # Common parameters
            self.penalty_factor = config.get('anti_cartel', 'penalty_factor')
            
            if self.mechanism_type == 'detection':
                # Detection-specific parameters
                self.monitoring_window = config.get('anti_cartel', 'monitoring_window')
                self.similarity_threshold = config.get('anti_cartel', 'similarity_threshold')
                
                # Enhanced price history tracking
                self.price_history = deque(maxlen=self.monitoring_window)
                self.current_episode_prices = []
                
                # New parameters for better detection
                self.min_price_variance = 1e-4  # Minimum acceptable price variance
                self.price_band_threshold = 0.05  # Maximum allowed price difference (5%)
                self.sustained_pattern_threshold = int(self.monitoring_window * 0.3)  # Need 30% of window for pattern
                
                print(f"Detection mechanism initialized with:")
                print(f"- Monitoring window: {self.monitoring_window}")
                print(f"- Similarity threshold: {self.similarity_threshold}")
                print(f"- Price band threshold: {self.price_band_threshold}")
            
            elif self.mechanism_type == 'ceiling':
                # Ceiling mechanism parameters
                # markup_limit now represents the minimum discount from grid price
                self.markup_limit = config.get('anti_cartel', 'markup_limit')  # e.g., 0.2 means 20% below grid price
                self.market_elasticity = config.get('anti_cartel', 'market_elasticity')
            
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
        Main penalty calculation method
        """
        num_houses = len(selling_prices)
        
        if not self.is_active():
            return [0.0] * num_houses
        
        if self.mechanism_type == 'detection':
            return self._calculate_detection_penalties(selling_prices)
        elif self.mechanism_type == 'ceiling':
            return self._calculate_ceiling_penalties(selling_prices, grid_price)
        
        return [0.0] * num_houses
    
    def _calculate_detection_penalties(self, selling_prices):
        """
        Enhanced cartel detection using multiple indicators:
        1. Price correlation between houses
        2. Price variance over time
        3. Price clustering analysis
        4. Sustained pattern detection
        """
        if len(self.price_history) < self.sustained_pattern_threshold:
            return [0.0] * len(selling_prices)
            
        penalties = [0.0] * len(selling_prices)
        price_history_array = np.array(list(self.price_history))
        num_houses = len(selling_prices)
        
        # 1. Analyze price movements and correlations
        for i in range(num_houses):
            for j in range(i + 1, num_houses):
                if i >= price_history_array.shape[1] or j >= price_history_array.shape[1]:
                    continue
                
                prices_i = price_history_array[-self.sustained_pattern_threshold:, i]
                prices_j = price_history_array[-self.sustained_pattern_threshold:, j]
                
                # Calculate price statistics
                mean_i = np.mean(prices_i)
                mean_j = np.mean(prices_j)
                std_i = np.std(prices_i)
                std_j = np.std(prices_j)
                
                # Check for suspicious patterns
                
                # Pattern 1: Nearly identical prices
                if abs(mean_i - mean_j) / max(mean_i, mean_j) < self.price_band_threshold:
                    price_matching_penalty = self.penalty_factor * 0.5
                    penalties[i] += price_matching_penalty * selling_prices[i]
                    penalties[j] += price_matching_penalty * selling_prices[j]
                
                # Pattern 2: Very low price variance
                if std_i < self.min_price_variance and std_j < self.min_price_variance:
                    low_variance_penalty = self.penalty_factor * 0.3
                    penalties[i] += low_variance_penalty * selling_prices[i]
                    penalties[j] += low_variance_penalty * selling_prices[j]
                
                # Pattern 3: Price correlation
                if std_i > 0 and std_j > 0:  # Only calculate correlation if there's variation
                    correlation = np.corrcoef(prices_i, prices_j)[0, 1]
                    if not np.isnan(correlation) and correlation > self.similarity_threshold:
                        correlation_penalty = self.penalty_factor * 0.4
                        penalties[i] += correlation_penalty * selling_prices[i]
                        penalties[j] += correlation_penalty * selling_prices[j]
        
        # 2. Analyze global price patterns
        if len(self.price_history) >= self.monitoring_window:
            recent_prices = price_history_array[-self.monitoring_window:]
            global_mean = np.mean(recent_prices, axis=1)
            global_std = np.std(recent_prices, axis=1)
            
            # Check for suspicious global patterns
            if np.mean(global_std) < self.min_price_variance * 2:
                # All houses moving together with very little variation
                global_penalty = self.penalty_factor * 0.2
                penalties = [p + global_penalty * sp for p, sp in zip(penalties, selling_prices)]
        
        return penalties
        
    def get_price_ceiling(self, grid_price):
        """
        Calculate the maximum allowed selling price based on current grid price.
        For oligopolistic control, the ceiling is set below grid price.
        
        Args:
            grid_price: Current grid price from the utility
        Returns:
            price_ceiling: Maximum allowed selling price (below grid price)
        """
        if not self.is_active() or self.mechanism_type != 'ceiling':
            return float('inf')
        
        # Calculate oligopolistic price ceiling
        # If markup_limit is 0.2, ceiling will be 80% of grid price
        price_ceiling = grid_price * (1 - self.markup_limit)
        
        # Ensure ceiling stays within reasonable bounds
        min_ceiling = grid_price * 0.5  # Never allow prices below 50% of grid price
        max_ceiling = grid_price * 0.95  # Never allow prices above 95% of grid price
        
        return min(max(price_ceiling, min_ceiling), max_ceiling)

    def _calculate_ceiling_penalties(self, selling_prices, grid_price):
        """
        Calculate penalties for houses selling above the oligopolistic price ceiling.
        
        Args:
            selling_prices: List of current selling prices for each house
            grid_price: Current grid price from the utility
        Returns:
            penalties: List of penalties for each house
        """
        penalties = [0.0] * len(selling_prices)
        price_ceiling = self.get_price_ceiling(grid_price)
        
        for i, price in enumerate(selling_prices):
            if price > price_ceiling:
                # Penalty proportional to amount above ceiling
                excess = price - price_ceiling
                # Higher penalty factor for oligopolistic ceiling violations
                penalties[i] = self.penalty_factor * excess * 1.5  # 50% higher penalties
                
        return penalties

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