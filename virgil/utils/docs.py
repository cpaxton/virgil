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
from typing import Optional, Dict
from pathlib import Path


def get_docs_directory() -> str:
    """Get the path to the docs directory."""
    # Get the virgil package directory
    virgil_dir = Path(__file__).parent.parent
    docs_dir = virgil_dir.parent / "docs"
    return str(docs_dir)


def get_available_docs() -> Dict[str, str]:
    """
    Get a dictionary of available documentation files.

    Returns:
        Dict mapping doc names to their file paths
    """
    docs_dir = get_docs_directory()
    available_docs = {}

    # Map friendly names to markdown files
    doc_mapping = {
        "discord": "discord.md",
        "models": "testing-models.md",
        "model-list": "model-quick-reference.md",
        "weather": "weather-api-setup.md",
        "install": "install.md",
    }

    for name, filename in doc_mapping.items():
        doc_path = os.path.join(docs_dir, filename)
        if os.path.exists(doc_path):
            available_docs[name] = doc_path

    return available_docs


def get_doc_content(doc_name: str) -> Optional[str]:
    """
    Get the content of a documentation file.

    Args:
        doc_name: The name of the doc (e.g., "discord", "models", "weather")

    Returns:
        The content of the doc file, or None if not found
    """
    available_docs = get_available_docs()

    if doc_name not in available_docs:
        return None

    try:
        with open(available_docs[doc_name], "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def format_doc_summary(doc_name: str, max_length: int = 2000) -> Optional[str]:
    """
    Get a formatted summary of a documentation file suitable for Discord.

    Args:
        doc_name: The name of the doc
        max_length: Maximum length of the summary (Discord has 2000 char limit per message)

    Returns:
        Formatted summary or None if not found
    """
    content = get_doc_content(doc_name)
    if content is None:
        return None

    # Remove markdown formatting that might not display well
    lines = content.split("\n")
    formatted_lines = []
    in_code_block = False

    for line in lines:
        # Skip horizontal rules and similar markdown
        if line.strip().startswith("---"):
            continue

        # Skip large images (they won't display)
        if line.strip().startswith("![") and ".png" in line:
            continue

        # Track code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            formatted_lines.append(line)
            continue

        # Skip some headers if too many
        if line.startswith("#") and len(formatted_lines) > 50:
            # Only keep important headers
            if line.count("#") <= 2:  # Keep only h1 and h2
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)

        # Limit total length
        total_length = len("\n".join(formatted_lines))
        if total_length > max_length:
            break

    result = "\n".join(formatted_lines)
    if len(result) > max_length:
        # Truncate and add indicator
        result = (
            result[: max_length - 100]
            + "\n\n*[Content truncated. See full docs in repository.]*"
        )

    return result


def list_available_docs() -> str:
    """
    Get a list of all available documentation topics.

    Returns:
        A formatted string listing available docs
    """
    available_docs = get_available_docs()

    doc_descriptions = {
        "discord": "Discord Setup Guide - How to set up and run the Friend Discord bot",
        "models": "Model Testing Guide - How to test different LLM backends with Virgil",
        "model-list": "Model Quick Reference - Quick list of all 66+ available models",
        "weather": "Weather API Setup - How to configure weather queries for Friend bot",
        "install": "Installation Guide - How to install Virgil",
    }

    lines = ["**Available documentation topics:**\n"]
    for name, path in available_docs.items():
        desc = doc_descriptions.get(name, "Documentation")
        lines.append(f"- `{name}` - {desc}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the functions
    print("Available docs:")
    docs = get_available_docs()
    for name, path in docs.items():
        print(f"  {name}: {path}")

    print("\n" + "=" * 50)
    print("Listing available docs:")
    print(list_available_docs())

    print("\n" + "=" * 50)
    print("Testing doc retrieval (discord):")
    content = format_doc_summary("discord", max_length=500)
    if content:
        print(content[:500] + "...")
