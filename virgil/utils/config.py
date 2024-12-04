import json
import yaml
import os

def load_config(config_path):
    """
    Load configuration from a file.
    Supports JSON and YAML formats.

    Args:
    config_path (str): Path to the configuration file.

    Returns:
    dict: Loaded configuration as a dictionary.

    Raises:
    FileNotFoundError: If the config file doesn't exist.
    ValueError: If the file format is not supported.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _, file_extension = os.path.splitext(config_path)

    with open(config_path, 'r') as config_file:
        if file_extension.lower() in ['.json']:
            return json.load(config_file)
        elif file_extension.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {file_extension}")

# Example usage
if __name__ == "__main__":
    try:
        # Assuming your config file is in the 'config' folder
        config = load_config('config/labyrinth.yaml')
        print("Loaded configuration:", config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}")

