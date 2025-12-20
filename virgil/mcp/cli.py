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

"""
Command-line interface for testing Virgil MCP server capabilities.
Allows sending messages to Discord, generating images, etc. via command line.
"""

import asyncio
import sys
import click
from typing import Optional
from virgil.services.image_service import VirgilImageService
from virgil.services.message_service import DiscordMessageService
from virgil.image import (
    DiffuserImageGenerator,
    FluxImageGenerator,
    QwenLayeredImageGenerator,
)
import discord


class DiscordClientContext:
    """Context manager for Discord client."""

    def __init__(self, token: Optional[str] = None):
        if token is None:
            import os
            from dotenv import load_dotenv

            load_dotenv()
            token = os.getenv("DISCORD_TOKEN")
            if not token:
                raise ValueError(
                    "DISCORD_TOKEN not found. Set it as environment variable or pass --token"
                )

        self.token = token
        self.client = None
        self.ready_event = asyncio.Event()

    async def __aenter__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            print(f"✓ Connected to Discord as {self.client.user}")
            self.ready_event.set()

        # Start client in background task
        async def run_client():
            try:
                await self.client.start(self.token)
            except Exception as e:
                print(f"✗ Failed to connect to Discord: {e}")
                self.ready_event.set()
                raise

        # Start the client
        asyncio.create_task(run_client())
        await self.ready_event.wait()

        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()


@click.group()
def cli():
    """Virgil MCP CLI - Test MCP server capabilities from command line."""
    pass


@cli.command()
@click.argument("message")
@click.option("--channel-id", help="Discord channel ID to send message to")
@click.option("--user-id", help="Discord user ID to send DM to")
@click.option("--token", help="Discord bot token (or set DISCORD_TOKEN env var)")
def send_discord_message(
    message: str,
    channel_id: Optional[str],
    user_id: Optional[str],
    token: Optional[str],
):
    """Send a message to Discord via MCP services.

    Example:
        python -m virgil.mcp.cli send-discord-message "Hello!" --channel-id 123456789
    """

    async def _send():
        if not channel_id and not user_id:
            print("✗ Error: Either --channel-id or --user-id must be provided")
            sys.exit(1)

        async with DiscordClientContext(token) as client:

            def get_channel(cid: int):
                return client.get_channel(cid)

            def get_user(uid: int):
                return client.get_user(uid)

            message_service = DiscordMessageService(
                discord_channel_getter=get_channel,
                discord_user_getter=get_user,
            )

            try:
                success = await message_service.send_message(
                    content=message,
                    channel_id=channel_id,
                    user_id=user_id,
                )
                if success:
                    print(f"✓ Message sent successfully: {message}")
                else:
                    print("✗ Failed to send message")
                    sys.exit(1)
            except Exception as e:
                print(f"✗ Error sending message: {e}")
                import traceback

                traceback.print_exc()
                sys.exit(1)

    try:
        asyncio.run(_send())
    except KeyboardInterrupt:
        print("\n✗ Interrupted")
        sys.exit(1)


@cli.command()
@click.argument("prompt")
@click.option("--output", "-o", default="generated_image.png", help="Output file path")
@click.option(
    "--image-generator",
    default="diffuser",
    type=click.Choice(["diffuser", "flux", "qwen-layered"]),
)
def generate_image(prompt: str, output: str, image_generator: str):
    """Generate an image from a text prompt.

    Example:
        virgil.mcp.cli generate-image "a beautiful sunset" -o sunset.png
    """
    # Initialize image generator
    if image_generator.lower() == "diffuser":
        image_gen = DiffuserImageGenerator(
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            model="turbo",
            xformers=False,
        )
    elif image_generator.lower() == "flux":
        image_gen = FluxImageGenerator(height=512, width=512)
    elif (
        image_generator.lower() == "qwen-layered"
        or image_generator.lower() == "qwen_layered"
    ):
        image_gen = QwenLayeredImageGenerator(
            height=640,
            width=640,
            num_inference_steps=50,
            layers=4,
            resolution=640,
        )
    else:
        print(f"Unknown image generator: {image_generator}")
        sys.exit(1)

    image_service = VirgilImageService(image_gen)

    try:
        print(f"Generating image from prompt: {prompt}")
        image = image_service.generate_image(prompt)
        image.save(output)
        print(f"✓ Image saved to: {output}")
    except Exception as e:
        print(f"✗ Error generating image: {e}")
        sys.exit(1)


@cli.command()
@click.argument("prompt")
@click.option("--channel-id", required=True, help="Discord channel ID to send image to")
@click.option("--caption", help="Optional caption for the image")
@click.option("--token", help="Discord bot token (or set DISCORD_TOKEN env var)")
@click.option(
    "--image-generator",
    default="diffuser",
    type=click.Choice(["diffuser", "flux", "qwen-layered"]),
)
def send_image(
    prompt: str,
    channel_id: str,
    caption: Optional[str],
    token: Optional[str],
    image_generator: str,
):
    """Generate and send an image to Discord.

    Example:
        virgil.mcp.cli send-image "a cat" --channel-id 123456789
    """

    async def _send():
        async with DiscordClientContext(token) as client:
            # Initialize image generator
            if image_generator.lower() == "diffuser":
                image_gen = DiffuserImageGenerator(
                    height=512,
                    width=512,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    model="turbo",
                    xformers=False,
                )
            elif image_generator.lower() == "flux":
                image_gen = FluxImageGenerator(height=512, width=512)
            else:
                print(f"Unknown image generator: {image_generator}")
                sys.exit(1)

            image_service = VirgilImageService(image_gen)

            def get_channel(cid: int):
                return client.get_channel(cid)

            def get_user(uid: int):
                return client.get_user(uid)

            message_service = DiscordMessageService(
                discord_channel_getter=get_channel,
                discord_user_getter=get_user,
            )

            try:
                print(f"Generating image from prompt: {prompt}")
                image = image_service.generate_image(prompt)

                print(f"Sending image to channel {channel_id}...")
                success = await message_service.send_image(
                    image=image,
                    channel_id=channel_id,
                    caption=caption,
                )

                if success:
                    print("✓ Image sent successfully")
                else:
                    print("✗ Failed to send image")
                    sys.exit(1)
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback

                traceback.print_exc()
                sys.exit(1)

    try:
        asyncio.run(_send())
    except KeyboardInterrupt:
        print("\n✗ Interrupted")
        sys.exit(1)


@cli.command()
@click.argument("message")
@click.option("--channel-id", required=True, help="Discord channel ID to post to")
@click.option("--token", help="Discord bot token (or set DISCORD_TOKEN env var)")
def post_in_channel(message: str, channel_id: str, token: Optional[str]):
    """Post a message in a Discord channel.

    Example:
        virgil.mcp.cli post-in-channel "Hello everyone!" --channel-id 123456789
    """

    async def _post():
        async with DiscordClientContext(token) as client:

            def get_channel(cid: int):
                return client.get_channel(cid)

            def get_user(uid: int):
                return client.get_user(uid)

            message_service = DiscordMessageService(
                discord_channel_getter=get_channel,
                discord_user_getter=get_user,
            )

            try:
                success = await message_service.post_in_channel(
                    content=message,
                    channel_id=channel_id,
                )
                if success:
                    print(f"✓ Message posted successfully: {message}")
                else:
                    print("✗ Failed to post message")
                    sys.exit(1)
            except Exception as e:
                print(f"✗ Error posting message: {e}")
                import traceback

                traceback.print_exc()
                sys.exit(1)

    try:
        asyncio.run(_post())
    except KeyboardInterrupt:
        print("\n✗ Interrupted")
        sys.exit(1)


@cli.command()
@click.argument("message")
@click.option("--user-id", required=True, help="Discord user ID to message")
@click.option("--token", help="Discord bot token (or set DISCORD_TOKEN env var)")
def message_user(message: str, user_id: str, token: Optional[str]):
    """Send a direct message to a Discord user.

    Example:
        virgil.mcp.cli message-user "Hello!" --user-id 123456789
    """

    async def _message():
        async with DiscordClientContext(token) as client:

            def get_channel(cid: int):
                return client.get_channel(cid)

            def get_user(uid: int):
                return client.get_user(uid)

            message_service = DiscordMessageService(
                discord_channel_getter=get_channel,
                discord_user_getter=get_user,
            )

            try:
                success = await message_service.message_user(
                    content=message,
                    user_id=user_id,
                )
                if success:
                    print(f"✓ Message sent successfully: {message}")
                else:
                    print("✗ Failed to send message")
                    sys.exit(1)
            except Exception as e:
                print(f"✗ Error sending message: {e}")
                import traceback

                traceback.print_exc()
                sys.exit(1)

    try:
        asyncio.run(_message())
    except KeyboardInterrupt:
        print("\n✗ Interrupted")
        sys.exit(1)


if __name__ == "__main__":
    cli()
