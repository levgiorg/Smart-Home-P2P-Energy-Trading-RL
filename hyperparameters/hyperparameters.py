import os
import json

class Config:
    config = None  # Class variable to store configuration
    config_file_path = None  # Class variable to store the path to the config file

    def __init__(self, config_file='hyperparameters.json'):
        if Config.config is None:
            # Determine the absolute path to the config file
            Config.config_file_path = os.path.join(os.path.dirname(__file__), config_file)
            
            # Check if the config file exists
            if not os.path.exists(Config.config_file_path):
                raise FileNotFoundError(f"Configuration file not found: {Config.config_file_path}")
            
            # Load the configuration from the JSON file
            with open(Config.config_file_path, 'r') as f:
                Config.config = json.load(f)
        # No need to set self.config; use Config.config directly

    def get(self, section, key):
        """Retrieve a specific value from the configuration."""
        return Config.config.get(section, {}).get(key)
        
    def get_section(self, section):
        """Retrieve an entire section from the configuration."""
        return Config.config.get(section, {})
        
    def set(self, section, option, value):
        """
        Set a configuration option and save the updated configuration back to the JSON file.
        
        Args:
            section (str): The section in the configuration.
            option (str): The option/key within the section.
            value: The value to set for the given option.
        """
        # Update the in-memory configuration
        if section in Config.config:
            Config.config[section][option] = value
        else:
            Config.config[section] = {option: value}
        
        # Save the updated configuration back to the JSON file
        try:
            with open(Config.config_file_path, 'w') as f:
                json.dump(Config.config, f, indent=4)
            print(f"Configuration updated successfully: [{section}] {option} = {value}")
        except IOError as e:
            print(f"Failed to write to configuration file: {e}")

