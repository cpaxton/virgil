from virgil.parser import Parser
from typing import List, Any, Optional, Tuple, Dict

from termcolor import colored

import re

def extract_tags(text: str, tags: List[str]) -> List[Tuple[str, str]]:
    result = []

    # Create a pattern that matches any of the given tags
    tag_pattern = '|'.join(map(re.escape, tags))
    pattern = f"<({tag_pattern})>(.*?)</\\1>"

    # Find all matches
    matches = re.finditer(pattern, text, re.DOTALL)

    # Extract and store matches in the order they appear
    for match in matches:
        tag = match.group(1)
        content = match.group(2).strip()
        result.append((tag, content))

    return result

class ChatbotActionParser(Parser):

    def parse(self, text: str): #  -> Optional[List[str, Any]]:
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


if __name__ == '__main__':
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
    """

    tags_to_extract = ["action", "thought", "observation"]
    result = extract_tags(text, tags_to_extract)

    for tag, content in result:
        print(f"{tag}: {content}")

