import os

def load_prompt(prompt_filename: str) -> str:
    """
    Load a prompt from a file. Will skip lines starting with '#'.

    Args:
        prompt_filename (str): Path to the prompt file.

    Returns:
        str: The loaded prompt text.
    """
    if os.path.isfile(prompt_filename):
        with open(prompt_filename, "r") as f:
            text = f.read()
            parsed_text = ""
            for line in text.splitlines():
                if line is None:
                    continue
                elif len(line) == 0:
                    parsed_text += "\n"
                elif len(line) > 0 and line.strip()[0] != "#":
                    parsed_text += line + "\n"
            return parsed_text
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_filename}")
