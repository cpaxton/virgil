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
                elif len(line) == 0 or len(line.strip()) == 0:
                    parsed_text += "\n"
                elif len(line) > 0 and line.strip()[0] != "#":
                    parsed_text += line + "\n"
            return parsed_text
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_filename}")
