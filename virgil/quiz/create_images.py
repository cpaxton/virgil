import os
import yaml
from typing import Dict, List
import click

from virgil.image.diffuser import DiffuserImageGenerator
from virgil.image.siglip import SigLIPAligner


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

    generator = DiffuserImageGenerator()
    aligner = SigLIPAligner()

    if len(folder_path) == 0:
        folder_path = "What sea creature are you?/2024-10-05-22-26-28/"

    # Create subfolders for the images for questions and for the results
    results_folder = os.path.join(folder_path, "results")
    questions_folder = os.path.join(folder_path, "questions")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(questions_folder, exist_ok=True)

    try:
        results, question_images = parse_yaml_files(folder_path)
        
        print()
        print("Results:")
        for letter, image in results.items():
            print("-" * 20)
            print(f"Letter: {letter}")
            print(f"Image: {image}")

            # Create the image using the DiffuserImageGenerator and search
            score, image = aligner.search(generator, image, num_tries=25)
            print(f"Final score: {score}")

            # Save the image
            image.save(os.path.join(results_folder, f"{letter}.png"))
            print(f"Image saved to {os.path.join(results_folder, f'{letter}.png')}")
            print()

        
        print("Question Images:")
        for i, image in enumerate(question_images, 1):
            print("-" * 20)
            print(f"Image {i}: {image}")

            # Create the image using the DiffuserImageGenerator and search
            score, image = aligner.search(generator, image, num_tries=25)
            print(f"Final score: {score}")

            # Save the image
            image.save(os.path.join(questions_folder, f"{i}.png"))
            print(f"Image saved to {os.path.join(questions_folder, f'{i}.png')}")
            
            # Print a newline
            print()
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
