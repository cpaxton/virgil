import yaml
import os
import click
import re


import pkg_resources
import os

def load_quiz_html():
    # Get the path to the quiz.html file
    file_path = pkg_resources.resource_filename('virgil.quiz', 'quiz.html')
    
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content

# Usage
quiz_html = load_quiz_html()
print(quiz_html)


def read_and_parse_yaml_files(results_file, questions_file):
    try:
        # Read the results.yaml file
        with open(results_file, 'r') as results_stream:
            results_data = yaml.safe_load(results_stream)

        # Read the questions.yaml file
        with open(questions_file, 'r') as questions_stream:
            questions_data = yaml.safe_load(questions_stream)

        # Parse and restructure the data
        parsed_questions = []
        for index, question in enumerate(questions_data, start=1):
            parsed_question = {
                "text": question["question"],
                "image": f"questions/{index}.png",
                "options": []
            }
            for key, value in question["options"].items():
                parsed_question["options"].append({
                    "text": value,
                    "type": key
                })
            parsed_questions.append(parsed_question)

        parsed_results = {}
        for key, value in results_data.items():
            if isinstance(value, dict):
                parsed_results[key] = {
                    "description": value["description"],
                    "image": f"results/{key}.png"
                }

        return {
            "questions": parsed_questions,
            "results": parsed_results
        }

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: YAML parsing error - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def create_combined_yaml_for_folder(folder_path: str) -> None:

    # Example usage
    results_file_path = os.path.join(folder_path, 'results.yaml')
    questions_file_path = os.path.join(folder_path, 'questions.yaml')

    parsed_data = read_and_parse_yaml_files(results_file_path, questions_file_path)

    if parsed_data:
        print(yaml.dump(parsed_data, default_flow_style=False))

    # Dump it to combined.yaml
    combined_file_path = os.path.join(folder_path, 'combined.yaml')
    with open(combined_file_path, 'w') as combined_stream:
        yaml.dump(parsed_data, combined_stream, default_flow_style=False)


def create_quiz_html(folder_path: str):
    """Create a quiz HTML file for the given folder containing results.yaml and questions.yaml.

    Args:
        folder_path (str): Path to the folder containing results.yaml and questions.yaml.
    """
    create_combined_yaml_for_folder(folder_path)

    # Load the combined.yaml file
    combined_file_path = os.path.join(folder_path, 'combined.yaml')
    with open(combined_file_path, 'r') as combined_stream:
        combined_data = yaml.safe_load(combined_stream)

    # Load the HTML template
    template = load_quiz_html()

    # Replace the placeholders in the template with the data from the combined.yaml file
    template.replace("$RAW_DATA", yaml.dump(combined_data, default_flow_style=False))
    print(template)

    # Write to file in the same folder
    output_file_path = os.path.join(folder_path, 'quiz.html')
    with open(output_file_path, 'w') as output_stream:
        output_stream.write(template)


# Example usage
@click.command()
@click.option("--folder_path", default="", help="Path to the folder containing results.yaml and questions.yaml")
def main(folder_path: str = ""):
    if len(folder_path) == 0:
        # folder_path = "What sea creature are you?/2024-10-05-22-26-28/"
        folder_path = "What houseplant are you?/2024-10-05-23-22-27/"
    create_quiz_html(folder_path)

if __name__ == "__main__":
    main()
