import requests
import os
import json
from datetime import datetime, timezone
import click

from .base import Social

import re
from typing import List, Dict


def parse_mentions(text: str) -> List[Dict]:
    """
    Code from https://docs.bsky.app/blog/create-post
    Parse mentions in a given text.

    Args:
        text (str): The text to parse for mentions.
    Returns:
        List[Dict]: A list of dictionaries containing the start and end indices of mentions and the handle.
    """
    spans = []
    # regex based on: https://atproto.com/specs/handle#handle-identifier-syntax
    mention_regex = rb"[$|\W](@([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)"
    text_bytes = text.encode("UTF-8")
    for m in re.finditer(mention_regex, text_bytes):
        spans.append(
            {
                "start": m.start(1),
                "end": m.end(1),
                "handle": m.group(1)[1:].decode("UTF-8"),
            }
        )
    return spans


def parse_urls(text: str) -> List[Dict]:
    """Parse URLs in a given text.
    Code from https://docs.bsky.app/blog/create-post

    Args:
        text (str): The text to parse for URLs.

    Returns:
        List[Dict]: A list of dictionaries containing the start and end indices of URLs and the URL itself.
    """
    spans = []
    # partial/naive URL regex based on: https://stackoverflow.com/a/3809435
    # tweaked to disallow some training punctuation
    url_regex = rb"[$|\W](https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*[-a-zA-Z0-9@%_\+~#//=])?)"
    text_bytes = text.encode("UTF-8")
    for m in re.finditer(url_regex, text_bytes):
        spans.append(
            {
                "start": m.start(1),
                "end": m.end(1),
                "url": m.group(1).decode("UTF-8"),
            }
        )
    return spans


class Bluesky(Social):
    def __init__(self, username, connection_env_var: str = "BLUESKY_APP_PASSWORD"):
        super().__init__(username)
        self.api_url = "https://api.bsky.app/xrpc/"

        self.username = username
        self.connection_env_var = connection_env_var

        # Load the app password from the environment variable
        self.app_password = os.getenv(self.connection_env_var)
        if not self.app_password:
            raise ValueError(
                f"Environment variable '{self.connection_env_var}' not set."
            )

        self.session = None

    def connect(self) -> None:
        """Connect using the stored app password."""
        resp = requests.post(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            json={"identifier": self.username, "password": self.app_password},
        )
        resp.raise_for_status()
        self.session = resp.json()
        print(self.session["accessJwt"])

    def post(self, content: str) -> dict:
        """
        Post content to Bluesky.
        Args:
            content (str): The content to post.
        Returns:
            dict: The response from the Bluesky API.
        """
        # Fetch the current time
        # Using a trailing "Z" is preferred over the "+00:00" format
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Required fields that each post must include
        post = {
            "$type": "app.bsky.feed.post",
            "text": content,
            "createdAt": now,
            "lang": "en-US",
        }

        resp = requests.post(
            "https://bsky.social/xrpc/com.atproto.repo.createRecord",
            headers={"Authorization": "Bearer " + self.session["accessJwt"]},
            json={
                "repo": self.session["did"],
                "collection": "app.bsky.feed.post",
                "record": post,
            },
        )
        print(json.dumps(resp.json(), indent=2))
        resp.raise_for_status()

        return resp.json()


@click.command()
@click.option(
    "--username", type=str, default="virgil-robot.bsky.social", help="Bluesky username."
)
@click.option("--post", type=str, default=None, help="Content to post on Bluesky.")
def main(post: str | None = None, username: str = "virgil-robot.bsky.social"):
    """Main function to run the social media posting."""
    if post is None:
        post = "Hello, world! This is a default post."

    # Example usage of the Social class
    # Replace with actual subclass implementation
    social_media = Bluesky(username=username)
    social_media.connect()
    if post is not None and len(post) > 0:
        social_media.post(post)


if __name__ == "__main__":
    main()
