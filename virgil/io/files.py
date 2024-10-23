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

# (c) 2024 by Chris Paxton

import yaml


def write_dict_to_file(file_path, dict_obj):
    with open(file_path, "w") as file:
        for key, value in dict_obj.items():
            file.write(f"{key}: {value}\n")


def read_dict_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        return {line.split(": ")[0]: line.split(": ")[1].strip() for line in lines}


def write_list_of_dicts_to_yaml(file_path, list_of_dicts):
    with open(file_path, "w") as file:
        yaml.dump(list_of_dicts, file)
