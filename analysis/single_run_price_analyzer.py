import pickle
import sys
from pathlib import Path

import numpy as np


def analyze_prices(run_path):
    pkl_path = Path(run_path) / "data" / "dqn__selling_prices.pkl"
    with open(pkl_path, "rb") as f:
        prices = pickle.load(f)

    print(f"Type of prices: {type(prices)}")
    if isinstance(prices, np.ndarray):
        print(f"Shape: {prices.shape}")
    elif isinstance(prices, list):
        print(f"Length: {len(prices)}")
        if len(prices) > 0:
            print(f"Type of first element: {type(prices[0])}")
            if isinstance(prices[0], np.ndarray):
                print(f"Shape of first: {prices[0].shape}")

    # Adjust based on common structures
    if isinstance(prices, list) and len(prices) > 0:
        # Assume list of episodes, each episode list of house prices (averages?)
        # Convert to numpy [episodes, houses]
        prices_array = np.array(prices)
        print(f"Prices array shape: {prices_array.shape}")

        # Compute std across houses per episode
        std_per_episode = np.std(prices_array, axis=1)

        # Average std over all episodes
        avg_std_all = np.mean(std_per_episode)

        # Average std over last 100 episodes
        last_100_std = (
            np.mean(std_per_episode[-100:]) if len(std_per_episode) >= 100 else avg_std_all
        )

        # Overall stats
        avg_price = np.mean(prices_array)
        min_price = np.min(prices_array)
        max_price = np.max(prices_array)

        print(f"Overall average price: {avg_price:.4f}")
        print(f"Overall min price: {min_price:.4f}")
        print(f"Overall max price: {max_price:.4f}")
        print(f"Average std dev per episode (all): {avg_std_all:.4f}")
        print(f"Average std dev per episode (last 100): {last_100_std:.4f}")
        threshold = 0.05
        print(
            f"Prices are {'similar' if last_100_std < threshold else 'diverse'} in last 100 episodes (threshold {threshold})"
        )
    else:
        print("Unexpected data structure")
        return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python single_run_price_analyzer.py <run_path>")
        sys.exit(1)
    analyze_prices(sys.argv[1])
