# # (c) 2024 by Chris Paxton

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
