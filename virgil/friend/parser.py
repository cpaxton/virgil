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

from virgil.parser import Parser
from typing import List, Any, Optional, Tuple

from termcolor import colored

import re


def extract_tags(text: str, tags: List[str], allow_unmatched: bool = True) -> List[Tuple[str, str]]:
    """Extracts specified tags and their content from the given text.

    Args:
        text (str): The text to extract tags from.
        tags (List[str]): A list of tags to extract.
        allow_unmatched (bool): Whether to allow unmatched opening tags.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the tag and its content.
    """
    result = []

    # Create a pattern that matches any of the given tags
    tag_pattern = "|".join(map(re.escape, tags))
    pattern = f"<({tag_pattern})>(.*?)</\\1>"

    # Find all matches
    matches = re.finditer(pattern, text, re.DOTALL)

    # Extract and store matches in the order they appear
    for match in matches:
        tag = match.group(1)
        content = match.group(2).strip()
        result.append((tag, content))

    if allow_unmatched:
        # Remove all matched content from the text
        text = re.sub(pattern, "", text, flags=re.DOTALL)

        # Check for an unmatched <tag> at the end
        text += "</end>"  # Add a closing tag for unmatched tags
        unmatched_pattern = f"<({tag_pattern})>(?!.*?</end>)"
        if re.search(unmatched_pattern, text):
            print(colored("Warning: Unmatched opening tag found.", "yellow"))
            matches = re.finditer(unmatched_pattern, text)
            for match in matches:
                tag = match.group(1)
                content = text[match.start() :].strip()
                # Remove initial tag from content
                content = content.replace(f"<{tag}>", "").strip()
            # remove </end> from content
            content = content.replace("</end>", "")
            result.append((tag, content))

    return result


class ChatbotActionParser(Parser):
    def parse(self, text: str):  #  -> Optional[List[str, Any]]:
        """Parse the action tags:
          <action>...</action>
          <remember>...</remember>
          <imagine>...</imagine>

        Args:
            text (str): The text to parse.

        Returns:
            Optional[List[str, Any]]: A list of parsed text. Each element is a tuple of (action, content).
        """
        tags_to_extract = ["say", "remember", "imagine"]
        extracted_tags = extract_tags(text, tags_to_extract)

        return extracted_tags


if __name__ == "__main__":
    # Example usage
    text = """
    <action>
      Print "Hello World"
    </action>

    Some other text

    <thought>Consider the implications</thought>

    <action>
      for i in range(5):
        print(i)
    </action>

    <observation>The loop printed numbers 0 to 4</observation>
    <observation>unmatched tag example
    """

    tags_to_extract = ["action", "thought", "observation"]
    result = extract_tags(text, tags_to_extract)

    for tag, content in result:
        print(f"{tag}: {content}")
