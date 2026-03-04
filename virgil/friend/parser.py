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

# Content that looks like instruction repetition (list of action names)
_GARBAGE_SAY_PATTERNS = (
    r"^,\s*<",  # Starts with ", <" (e.g. ", <imagine>, <meme>")
    r"^<imagine>\s*,\s*<meme>",  # List of action tags
    r"^<say>\s*,\s*<imagine>",  # Repeated from instructions
)


def _is_garbage_say_content(content: str) -> bool:
    """Return True if say content looks like instruction repetition, not a real message."""
    if not content or len(content.strip()) < 3:
        return True
    content_stripped = content.strip()
    for pattern in _GARBAGE_SAY_PATTERNS:
        if re.search(pattern, content_stripped):
            return True
    # Content that's mostly action tag names
    if re.search(r"^[<\s,\w>]+$", content_stripped) and "<" in content_stripped:
        return True
    return False


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
        # Also remove any unclosed <think> tags at the start
        text = re.sub(r"^.*?<think>", "", text, flags=re.DOTALL)
        # Remove any remaining unclosed <think> tags
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        # Strip <thought>...</thought> (model sometimes uses this instead of <think>)
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
        # Remove any content before the first real action tag (handles models that
        # output instruction repetition like "actions: <say>, <imagine>, ...")
        # Require content to start with a letter (skip ", <imagine>", "<meme>." etc.)
        first_action = re.search(
            r"<(say|imagine|meme|remember|forget|weather|remind|schedule|help)>\s*[A-Za-z]",
            text,
            re.DOTALL,
        )
        if first_action:
            text = text[first_action.start() :]

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

        # Filter garbage: "say" content that looks like instruction repetition
        # (e.g. ", <imagine>, <meme>, ..." from "actions: <say>, <imagine>, ...")
        if tag == "say" and _is_garbage_say_content(content):
            continue

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

                # Filter garbage say content from unmatched tags too
                if tag == "say" and _is_garbage_say_content(content):
                    continue

                result.append((tag, content, attributes))

    # if nothing at all was parsed, try to extract <say> content or use fallback
    if len(result) == 0:
        # Last resort: extract <say>...</say> even from malformed output
        say_match = re.search(r"<say>(.*?)</say>", text, re.DOTALL)
        if say_match:
            content = say_match.group(1).strip()
            if not _is_garbage_say_content(content):
                result.append(("say", content, {}))
        if len(result) == 0:
            # remove any dangling tags and try fallback
            text = re.sub(r"<[^>]+>", "", text)
            clean = text.strip()
            # Don't send structured reasoning or instruction repetition as a message
            if (
                clean
                and not _is_garbage_say_content(clean)
                and not re.search(
                    r"^(Analyze|Determine|Draft|Final|Action|Required|1\.|2\.)",
                    clean,
                    re.IGNORECASE,
                )
            ):
                result.append(("say", clean, {}))

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
            "meme",
            "think",
            "weather",
            "edit_image",
            "remind",
            "show_remind",
            "edit_remind",
            "delete_remind",
            "schedule",
            "show_schedule",
            "unschedule",
            "edit_schedule",
            "help",
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
