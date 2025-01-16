import os
import json
import shutil
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

class BookKeeper:
    def __init__(self, config, model_name='ddpg_', mode='train', run_dir=None):
        # Initialize run directory
        runs_dir = 'runs'
        os.makedirs(runs_dir, exist_ok=True)
        if run_dir is None:
            # Get list of existing run_X directories
            existing_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith('run_')]
            run_numbers = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
            if run_numbers:
                next_run_number = max(run_numbers) + 1
            else:
                next_run_number = 1
            self.run_dir = os.path.join(runs_dir, f'run_{next_run_number}')
        else:
            self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create subdirectory for mode (train or evaluate)
        self.mode = mode
        self.mode_dir = os.path.join(self.run_dir, mode)
        os.makedirs(self.mode_dir, exist_ok=True)

        # Save hyperparameters
        hyperparameters_source_path = 'hyperparameters/hyperparameters.json'
        hyperparameters_dest_path = os.path.join(self.run_dir, 'hyperparameters.json')
        if os.path.exists(hyperparameters_source_path):
            shutil.copy(hyperparameters_source_path, hyperparameters_dest_path)
        else:
            print('hyperparameters.json not found.')

        # Initialize metrics storage
        self.metrics = {
            'scores': [],
            'rewards_per_house': [],
            'temperatures': [],
            'HVAC_energy_cons': [],
            'depreciation': [],
            'penalty': []
        }

        self.model_name = model_name

        # Output directory for data files
        self.output_dir = os.path.join(self.mode_dir, 'data')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def log_episode(self, episode, score, rewards_per_house=None, temperatures_episode=None, HVAC_energy_cons_per_house=None, depreciation_per_house=None, penalty_per_house=None):
        # Log per-episode metrics
        self.metrics['scores'].append(score)
        if rewards_per_house is not None:
            self.metrics['rewards_per_house'].append(rewards_per_house)
        if temperatures_episode is not None:
            self.metrics['temperatures'].append(temperatures_episode)
        if HVAC_energy_cons_per_house is not None:
            self.metrics['HVAC_energy_cons'].append(HVAC_energy_cons_per_house)
        if depreciation_per_house is not None:
            self.metrics['depreciation'].append(depreciation_per_house)
        if penalty_per_house is not None:
            self.metrics['penalty'].append(penalty_per_house)
    
    def save_metrics(self):
        # Save metrics to files
        # Save scores
        with open(os.path.join(self.output_dir, self.model_name + '_rewards.pkl'), 'wb') as f:
            pickle.dump(self.metrics['scores'], f)
        # Save temperatures if available
        if self.metrics['temperatures']:
            with open(os.path.join(self.output_dir, self.model_name + '_temperatures.pkl'), 'wb') as f:
                pickle.dump(self.metrics['temperatures'], f)
        # Save other metrics if available
        if self.metrics['HVAC_energy_cons']:
            with open(os.path.join(self.output_dir, self.model_name + '_HVAC_energy_cons.pkl'), 'wb') as f:
                pickle.dump(self.metrics['HVAC_energy_cons'], f)
        if self.metrics['depreciation']:
            with open(os.path.join(self.output_dir, self.model_name + '_depreciation.pkl'), 'wb') as f:
                pickle.dump(self.metrics['depreciation'], f)
        if self.metrics['penalty']:
            with open(os.path.join(self.output_dir, self.model_name + '_penalty.pkl'), 'wb') as f:
                pickle.dump(self.metrics['penalty'], f)
    
    def plot_metrics(self, model_label=None, plot_average_only=False):
        """
        Plot various metrics and save the plots.

        Parameters:
        - model_label (str): Optional label for the model to be used in plot titles.
        - plot_average_only (bool): If True, plot only the average lines.
        """
        # Determine the number of episodes
        num_episodes = len(self.metrics['scores'])
        episodes = range(1, num_episodes + 1)

        # Use the model_label in plot titles if provided
        if model_label is None:
            model_label = self.model_name

        if 'centralized' in self.model_name:
            mode_label = 'Centralized Mode'
        elif 'decentralized' in self.model_name:
            mode_label = 'Decentralized Mode'
        else:
            mode_label = ''

        # Define markers and color map for consistency across all plots
        markers = ['o', 's', '^', 'D', '*', 'p', 'x', '+', 'v', '<', '>']

        # Check if per-house rewards are available (decentralized mode)
        decentralized_mode = self.metrics.get('rewards_per_house') and len(self.metrics['rewards_per_house'][0]) > 1

        # Plotting function for centralized mode
        def plot_centralized(metric_key, ylabel, title, filename):
            data = [value[0] if isinstance(value, list) else value for value in self.metrics[metric_key]]
            average_data = np.cumsum(data) / np.arange(1, num_episodes + 1)

            plt.figure(figsize=(12, 8))
            if not plot_average_only:
                plt.plot(episodes, data, label=f'{ylabel} per Episode', linestyle='-', color='blue', linewidth=1)
            plt.plot(episodes, average_data, label=f'Average {ylabel}', linestyle='--', color='red', linewidth=2)
            plt.xlabel('Episode', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.title(f'{title} - {model_label}', fontsize=16)
            plt.legend(fontsize='small', loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plot_path = os.path.join(self.mode_dir, filename)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

        # Plotting function for decentralized mode
        def plot_decentralized(metric_key, ylabel, title, filename):
            data_per_house = list(zip(*self.metrics[metric_key]))
            num_houses = len(data_per_house)
            color_map = cm.get_cmap('tab10', num_houses)

            plt.figure(figsize=(12, 8))
            for house_idx in range(num_houses):
                house_data = np.array(data_per_house[house_idx])
                average_data = np.cumsum(house_data) / np.arange(1, num_episodes + 1)
                color = color_map(house_idx)
                marker = markers[house_idx % len(markers)]

                if not plot_average_only:
                    plt.plot(
                        episodes,
                        house_data,
                        label=f'House {house_idx}',
                        marker=marker,
                        linestyle='-',
                        color=color,
                        markevery=5,
                        linewidth=1
                    )
                
                plt.plot(
                    episodes,
                    average_data,
                    label=f'House {house_idx} Avg' if not plot_average_only else f'House {house_idx} Average',
                    linestyle='--',
                    color=color,
                    linewidth=2
                )

            plt.xlabel('Episode', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.title(f'{title} - {model_label}', fontsize=16)
            plt.legend(fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plot_path = os.path.join(self.mode_dir, filename)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

        # Conditionally plot total rewards only if not in decentralized mode
        if not decentralized_mode:
            # Plot total rewards
            scores = self.metrics['scores']
            average_rewards = np.cumsum(scores) / np.arange(1, num_episodes + 1)

            plt.figure(figsize=(12, 8))
            if not plot_average_only:
                plt.plot(
                    episodes,
                    scores,
                    label='Total Reward per Episode',
                    linestyle='-',
                    color='blue',
                    linewidth=1
                )
            plt.plot(
                episodes,
                average_rewards,
                label='Average Total Reward',
                linestyle='--',
                color='red',
                linewidth=2
            )
            plt.xlabel('Episode', fontsize=14)
            plt.ylabel('Total Reward', fontsize=14)
            plt.title(f'Total Rewards - {model_label}', fontsize=16)
            plt.legend(fontsize='small', loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plot_path = os.path.join(self.mode_dir, 'total_rewards_plot.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

        if decentralized_mode:
            # Plot per-house rewards
            plot_decentralized(
                metric_key='rewards_per_house',
                ylabel='Reward',
                title='Rewards per House',
                filename='rewards_per_house_plot.png'
            )

            # Plot other per-house metrics
            if self.metrics.get('HVAC_energy_cons'):
                plot_decentralized(
                    metric_key='HVAC_energy_cons',
                    ylabel='Total HVAC Energy Consumption',
                    title='HVAC Energy Consumption per Episode',
                    filename='HVAC_energy_consumption.png'
                )

            if self.metrics.get('depreciation'):
                plot_decentralized(
                    metric_key='depreciation',
                    ylabel='Total Battery Depreciation',
                    title='Battery Depreciation per Episode',
                    filename='battery_depreciation.png'
                )

            if self.metrics.get('penalty'):
                plot_decentralized(
                    metric_key='penalty',
                    ylabel='Total Temperature Deviation Penalty',
                    title='Temperature Deviation Penalty per Episode',
                    filename='temperature_deviation_penalty.png'
                )
        else:
            # Plot overall metrics
            if self.metrics.get('HVAC_energy_cons'):
                plot_centralized(
                    metric_key='HVAC_energy_cons',
                    ylabel='Total HVAC Energy Consumption',
                    title='HVAC Energy Consumption',
                    filename='HVAC_energy_consumption.png'
                )

            if self.metrics.get('depreciation'):
                plot_centralized(
                    metric_key='depreciation',
                    ylabel='Total Battery Depreciation',
                    title='Battery Depreciation',
                    filename='battery_depreciation.png'
                )

            if self.metrics.get('penalty'):
                plot_centralized(
                    metric_key='penalty',
                    ylabel='Total Temperature Deviation Penalty',
                    title='Temperature Deviation Penalty',
                    filename='temperature_deviation_penalty.png'
                )