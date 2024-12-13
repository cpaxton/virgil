# Copyright 2024 Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    with open(config_path, "r") as config_file:
        if file_extension.lower() in [".json"]:
            return json.load(config_file)
        elif file_extension.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {file_extension}")


# Example usage
if __name__ == "__main__":
    try:
        # Assuming your config file is in the 'config' folder
        config = load_config("config/labyrinth.yaml")
        print("Loaded configuration:", config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}")
