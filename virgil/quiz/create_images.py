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
    results_path = os.path.join(folder_path, "results.yaml")
    questions_path = os.path.join(folder_path, "questions.yaml")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.yaml not found in {folder_path}")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"questions.yaml not found in {folder_path}")

    # Parse results.yaml
    with open(results_path, "r") as file:
        results_data = yaml.safe_load(file)

    results = {}
    for key, value in results_data.items():
        if isinstance(value, dict) and "letter" in value and "image" in value:
            results[value["letter"]] = value["image"]

    # Parse questions.yaml
    with open(questions_path, "r") as file:
        questions_data = yaml.safe_load(file)

    question_images = []
    for item in questions_data:
        if isinstance(item, dict) and "image" in item:
            question_images.append(item["image"])

    return results, question_images


def create_images_for_folder(folder_path: str, num_tries: int = 10) -> None:
    """Create images for the given folder containing results.yaml and questions.yaml.

    Args:
        folder_path (str): Path to the folder containing results.yaml and questions.yaml.
    """
    # generator = DiffuserImageGenerator(num_inference_steps=50, guidance_scale=7.5)
    generator = DiffuserImageGenerator(
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=0.0,
        model="turbo",
        xformers=False,
    )
    aligner = SigLIPAligner()

    # Create subfolders for the images for questions and for the results
    results_folder = os.path.join(folder_path, "results")
    questions_folder = os.path.join(folder_path, "questions")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(questions_folder, exist_ok=True)

    try:
        results, question_images = parse_yaml_files(folder_path)

        print("Question Images:")
        for i, image in enumerate(question_images, 1):
            print("-" * 20)
            print(f"Image {i}: {image}")

            # Create the image using the DiffuserImageGenerator and search
            score, image = aligner.search(generator, image, num_tries=num_tries)
            print(f"Final score: {score}")

            # Save the image
            image.save(os.path.join(questions_folder, f"{i}.png"))
            print(f"Image saved to {os.path.join(questions_folder, f'{i}.png')}")

            # Print a newline
            print()

        print()
        print("Results:")
        for letter, image in results.items():
            print("-" * 20)
            print(f"Letter: {letter}")
            print(f"Image: {image}")

            # Create the image using the DiffuserImageGenerator and search
            score, image = aligner.search(generator, image, num_tries=num_tries)
            print(f"Final score: {score}")

            # Save the image
            image.save(os.path.join(results_folder, f"{letter}.png"))
            print(f"Image saved to {os.path.join(results_folder, f'{letter}.png')}")
            print()

    except FileNotFoundError as e:
        print(f"Error: {e}")


# Example usage
@click.command()
@click.option(
    "--folder_path",
    default="",
    help="Path to the folder containing many different quizzes",
)
@click.option(
    "--quiz_path",
    default="",
    help="Path to the folder containing the quiz results.yaml and questions.yaml",
)
def main(folder_path: str = "", quiz_path: str = ""):
    import glob

    if len(quiz_path) > 0:
        print("Try to create images for quiz:", quiz_path)
        create_images_for_folder(quiz_path)
        return
    elif len(folder_path) == 0:
        # folder_path = "What sea creature are you?/2024-10-05-22-26-28/"
        folder_path = "2024-10-07-v2"

    """
        # topics = ["What kind of cocktail are you?", "What kind of beer are you?"]
        topics = ["What kind of tooth are you?", "Which bone are you?", "What halloween costume are you?", "What halloween creature are you?", "Which day in October are you?", "What halloween candy are you?",
    # Gemma failed to generate a quiz for "what halloween creature are you?"
"Which cosmic horror are you devoted to?", "To which of the elder gods should you pray?", "Which afterlife will you end up in?",
"Which kind of undead monstrosity will you be?", "What holiday are you?", "What kind of door are you?", "What extremely specific door are you?",
                "What kind of wine are you?", "What kind of spirit are you?", "What kind of non-alcoholic beverage are you?", "What kind of juice are you?", "What kind of soda are you?"]
    for topic in topics:
        folder = os.path.join(folder_path, topic)
        print("Try to create images for folder:", folder)
        create_images_for_folder(folder)
    """

    all_folders = glob.glob(f"{folder_path}/*")
    for folder in all_folders:
        print("Try to create images for folder:", folder)
        create_images_for_folder(folder)


if __name__ == "__main__":
    main()
