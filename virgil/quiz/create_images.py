import os
import yaml
from typing import Dict, List
import click

def parse_yaml_files(folder_path: str) -> tuple[Dict[str, str], List[str]]:
    """
    Parse the results.yaml and questions.yaml files in the given folder.

    Args:
        folder_path (str): Path to the folder containing results.yaml and questions.yaml

    Returns:
        tuple: A tuple containing:
            - Dict[str, str]: A dictionary where the key is the letter and the value is the image description from results.yaml
            - List[str]: A list of image descriptions from questions.yaml
    """
    results_path = os.path.join(folder_path, 'results.yaml')
    questions_path = os.path.join(folder_path, 'questions.yaml')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.yaml not found in {folder_path}")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"questions.yaml not found in {folder_path}")

    # Parse results.yaml
    with open(results_path, 'r') as file:
        results_data = yaml.safe_load(file)

    results = {}
    for key, value in results_data.items():
        if isinstance(value, dict) and 'letter' in value and 'image' in value:
            results[value['letter']] = value['image']

    # Parse questions.yaml
    with open(questions_path, 'r') as file:
        questions_data = yaml.safe_load(file)

    question_images = []
    for item in questions_data:
        if isinstance(item, dict) and 'image' in item:
            question_images.append(item['image'])

    return results, question_images

# Example usage
@click.command()
@click.option("--folder_path", default="", help="Path to the folder containing results.yaml and questions.yaml")
def main(folder_path: str = ""):
    if len(folder_path) == 0:
        folder_path = "What sea creature are you?/2024-10-05-22-26-28/"

    try:
        results, question_images = parse_yaml_files(folder_path)
        
        print("Results:")
        for letter, image in results.items():
            print(f"Letter: {letter}")
            print(f"Image: {image}")
            print()
        
        print("Question Images:")
        for i, image in enumerate(question_images, 1):
            print(f"Image {i}: {image}")
            print()
    except FileNotFoundError as e:
        print(f"Error: {e}")