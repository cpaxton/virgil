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
from typing import List, Tuple

from termcolor import colored

import re


def extract_tags(
    text: str,
    tags: List[str],
    allow_unmatched: bool = True,
    prune_thoughts: bool = True,
) -> List[Tuple[str, str, dict]]:
    """Extracts specified tags and their content from the given text.

    Args:
        text (str): The text to extract tags from.
        tags (List[str]): A list of tags to extract.
        allow_unmatched (bool): Whether to allow unmatched opening tags.
        prune_thoughts (bool): Whether to ignore all tags before the first </think> tag.

    Returns:
        List[Tuple[str, str, dict]]: A list of tuples containing (tag, content, attributes).
        Attributes is a dict of attribute name-value pairs.
    """
    result = []

    # Create a pattern that matches any of the given tags with optional attributes
    tag_pattern = "|".join(map(re.escape, tags))
    # Pattern matches: <tag attr="value">content</tag> or <tag>content</tag>
    pattern = f"<({tag_pattern})([^>]*)>(.*?)</\\1>"

    if prune_thoughts:
        # Remove all content before the first </think> tag
        text = re.sub(r".*?</think>", "", text, flags=re.DOTALL)

    # Find all matches
    matches = re.finditer(pattern, text, re.DOTALL)

    # Extract and store matches in the order they appear
    for match in matches:
        tag = match.group(1)
        attr_string = match.group(2).strip()
        content = match.group(3).strip()

        # Parse attributes (e.g., time="12:31:00" or time='12:31:00')
        attributes = {}
        if attr_string:
            # Match attribute="value" or attribute='value'
            attr_pattern = r'(\w+)=["\']([^"\']+)["\']'
            attr_matches = re.findall(attr_pattern, attr_string)
            for attr_name, attr_value in attr_matches:
                attributes[attr_name] = attr_value

        result.append((tag, content, attributes))

    if allow_unmatched:
        # Remove all matched content from the text
        text = re.sub(pattern, "", text, flags=re.DOTALL)

        # Check for an unmatched <tag> at the end
        text += "</end>"  # Add a closing tag for unmatched tags
        unmatched_pattern = f"<({tag_pattern})([^>]*)>(?!.*?</end>)"
        if re.search(unmatched_pattern, text):
            print(colored("Warning: Unmatched opening tag found.", "yellow"))
            matches = re.finditer(unmatched_pattern, text)
            for match in matches:
                tag = match.group(1)
                attr_string = match.group(2).strip()
                content = text[match.start() :].strip()
                # Remove initial tag from content
                content = content.replace(f"<{tag}", "").strip()
                # Remove attributes if present
                if attr_string:
                    content = re.sub(r"[^>]*>", "", content, count=1)
                else:
                    content = content.replace(">", "", 1)
                # remove </end> from content
                content = content.replace("</end>", "")

                # always remove dangling tags in content
                content = re.sub(r"<[^>]+>", "", content)

                # Parse attributes for unmatched tag
                attributes = {}
                if attr_string:
                    attr_pattern = r'(\w+)=["\']([^"\']+)["\']'
                    attr_matches = re.findall(attr_pattern, attr_string)
                    for attr_name, attr_value in attr_matches:
                        attributes[attr_name] = attr_value

                result.append((tag, content, attributes))

    # if nothing at all was parsed, just say whatever was in the text
    if len(result) == 0:
        # remove any dangling tags
        text = re.sub(r"<[^>]+>", "", text)

        result.append(("say", text.strip(), {}))

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
        tags_to_extract = [
            "say",
            "remember",
            "imagine",
            "think",
            "weather",
            "edit_image",
            "remind",
            "schedule",
        ]
        extracted_tags = extract_tags(text, tags_to_extract, prune_thoughts=True)

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
