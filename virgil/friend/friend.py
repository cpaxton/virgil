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

import click
import os
import signal
from virgil.io.discord_bot import DiscordBot, Task
from virgil.backend import get_backend
from virgil.chat import ChatWrapper
from typing import Optional
import random
import threading
import time
from termcolor import colored
import io
import discord
from PIL import Image

# This only works on Ampere+ GPUs
import torch

# Enable TensorFloat-32 (TF32) on Ampere GPUs for faster matrix multiplications
# Need to let Ruff know this is okay
# ruff: noqa: F401
# ruff: noqa: E402
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from virgil.friend.parser import ChatbotActionParser
from virgil.friend.reminder import ReminderManager, Reminder
from virgil.friend.scheduler import Scheduler, ScheduledTask
from virgil.image import (
    DiffuserImageGenerator,
    FluxImageGenerator,
    QwenLayeredImageGenerator,
)
from virgil.utils import load_prompt
from virgil.utils.weather import (
    get_current_weather,
    validate_api_key,
    format_weather_message,
)
from virgil.services.image_service import VirgilImageService
from virgil.services.message_service import DiscordMessageService
from virgil.mcp.server import VirgilMCPServer
import asyncio
from datetime import datetime, timedelta
from virgil.utils.docs import (
    list_available_docs,
    format_doc_summary,
    get_available_docs,
)


def load_prompt_helper(prompt_filename: str = "prompt.txt") -> str:
    """Load the prompt from the given filename.

    Args:
        prompt_filename (str): The filename for the prompt. Defaults to "prompt.txt".

    Returns:
        str: The prompt as a string.
    """
    if prompt_filename[0] != "/":
        file_path = os.path.join(os.path.dirname(__file__), prompt_filename)
    else:
        file_path = prompt_filename

    return load_prompt(file_path)


class Friend(DiscordBot):
    """Friend is a simple discord bot, which chats with you if you are on its server.

    Be patient with it, it's very stupid. It will say goodbye in all active
    channels when it is shut down via Ctrl+C.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        backend="gemma",
        attention_window_seconds: float = 600.0,
        image_generator: str = "diffuser",
        join_at_random: bool = False,
        max_history_length: int = 25,
        prompt_filename: str = "prompt.txt",
        home_channel: str = "ask-a-robot",
        weather_api_key: Optional[str] = None,
        enable_mcp: bool = False,
    ) -> None:
        """Initialize the bot with the given token and backend.

        Args:
            token (Optional[str]): The token for the discord bot. Defaults to None.
            backend (str): The backend to use for the chat. Defaults to "gemma".
            attention_window_seconds (float): The number of seconds to pay attention to a channel. Defaults to 600.0.
            image_generator (str): The image generator to use ("diffuser", "flux", or "qwen-layered"). Defaults to "diffuser".
            join_at_random (bool): Whether to join channels at random. Defaults to False.
            max_history_length (int): The maximum length of the chat history. Defaults to 25.
            prompt_filename (str): The filename for the prompt. Defaults to "prompt.txt".
            home_channel (str): The name of the home channel to join. Defaults to "ask-a-robot".
            weather_api_key (Optional[str]): OpenWeatherMap API key for weather functionality. Defaults to None.
            enable_mcp (bool): Whether to enable MCP server alongside Discord bot. Defaults to False.
        """

        self.backend = get_backend(backend)
        self.chat = ChatWrapper(
            self.backend, max_history_length=max_history_length, preserve=2
        )
        self.attention_window_seconds = attention_window_seconds
        self.raw_prompt = load_prompt_helper(prompt_filename)
        self.prompt = None
        self._user_name = None
        self._user_id = None
        self.sent_prompt = False
        self.join_at_random = join_at_random
        self.home_channel = home_channel
        self.weather_api_key = weather_api_key
        self._weather_api_key_valid = False

        # Validate weather API key if provided
        if self.weather_api_key:
            print("Validating weather API key...")
            self._weather_api_key_valid = validate_api_key(self.weather_api_key)
            if self._weather_api_key_valid:
                print("Weather API key is valid.")
            else:
                print(
                    colored(
                        "Warning: Weather API key appears to be invalid. Weather functionality will be disabled.",
                        "yellow",
                    )
                )
        else:
            print(
                "No weather API key provided. Weather functionality will be disabled."
            )

        super(Friend, self).__init__(token)

        # Check to see if memory file exists
        # Memory is stored as a text file with a list of messages
        # Each message is just a string
        # The file is stored in the same directory as the bot
        # The file is named "memory.txt"
        memory_file = "memory.txt"
        # If the file does not exist, create it
        if not os.path.exists(memory_file):
            with open(memory_file, "w") as file:
                file.write("")
            memory = []
        else:
            # If the file does exist, load the memory into memory
            with open(memory_file, "r") as file:
                memory = file.read().split("\n")

        # Loaded memory
        self.memory = memory

        if isinstance(image_generator, str):
            if image_generator.lower() == "diffuser":
                # This worked well as of 2024-10-22 with the diffusers library
                # self.image_generator = DiffuserImageGenerator(height=512, width=512, num_inference_steps=20, guidance_scale=0.0, model="turbo", xformers=False)
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
            image_gen = image_generator

        # Keep image_generator for backward compatibility
        self.image_generator = image_gen

        # Initialize services
        self.image_service = VirgilImageService(image_gen)

        # Initialize message service with Discord-specific functions
        # Note: We'll set this up after the client is ready in on_ready()
        self.message_service = None

        self.parser = ChatbotActionParser(self.chat)

        self._chat_lock = threading.Lock()  # Lock for chat access

        # MCP server support
        self.enable_mcp = enable_mcp
        self.mcp_server = None
        self._mcp_task = None

        # Reminder system
        self.reminder_manager = ReminderManager()
        self.reminder_manager.set_execution_callback(self._execute_reminder)

        # Scheduler system
        self.scheduler = Scheduler()
        self.scheduler.set_execution_callback(self._execute_scheduled_task)

        # Shutdown handling
        self._shutdown_handler_setup = False

        # Note: Goodbye messages are handled in the run() method's KeyboardInterrupt handler
        # The monkey-patch approach was unreliable, so we handle it explicitly in shutdown

    async def say_goodbye(self):
        """Sends a goodbye message to all channels the bot is active in.

        This method is called automatically when the bot is shut down via Ctrl+C,
        ensuring a graceful exit. It iterates through all channels in the
        `allowed_channels` list and sends a final message.
        """
        print("Saying goodbye to active channels...")
        goodbye_message = "Goodbye! 👋"

        if not self.client or not self.client.is_ready() or self.client.is_closed():
            print("Client not ready or closed, skipping goodbye messages")
            return

        channels_to_notify = []
        # Collect all valid channels from allowed_channels
        for channel in self.allowed_channels:
            # Double-check channel is still valid and we have permission
            if channel and channel in self.allowed_channels:
                try:
                    # Check if we can send messages to this channel
                    if hasattr(channel, "guild") and channel.guild:
                        if channel.permissions_for(channel.guild.me).send_messages:
                            channels_to_notify.append(channel)
                except Exception:
                    # Skip channels we can't check permissions for
                    pass

        sent_count = 0
        for channel in channels_to_notify:
            try:
                # Check if client is still connected
                if self.client.is_closed():
                    break
                await channel.send(goodbye_message)
                print(f"Sent goodbye to {channel.name}")
                sent_count += 1
            except (
                discord.errors.HTTPException,
                discord.errors.ConnectionClosed,
            ) as e:
                # Session closed or connection error - stop trying
                print(f"Connection closed, stopping goodbye messages: {e}")
                break
            except Exception as e:
                print(f"Error sending goodbye to {channel.name}: {e}")

        if sent_count > 0:
            print(f"Sent goodbye messages to {sent_count} channel(s)")

    async def on_ready(self):
        """Event listener called when the bot has switched from offline to online."""
        print(f"{self.client.user} has connected to Discord!")
        guild_count = 0

        print("Bot User name:", self.client.user.name)
        print("Bot Global name:", self.client.user.global_name)
        print("Bot User IDL", self.client.user.id)
        self._user_name = self.client.user.name
        self._user_id = self.client.user.id

        # Initialize message service now that client is ready
        def get_channel(channel_id: int):
            return self.client.get_channel(channel_id)

        def get_user(user_id: int):
            return self.client.get_user(user_id)

        self.message_service = DiscordMessageService(
            discord_channel_getter=get_channel,
            discord_user_getter=get_user,
        )

        # Initialize MCP server if enabled
        if self.enable_mcp:
            try:
                self.mcp_server = VirgilMCPServer(
                    image_service=self.image_service,
                    message_service=self.message_service,
                )
                print("MCP server initialized successfully.")
                # Start MCP server in background
                self._start_mcp_server()
            except Exception as e:
                print(colored(f"Failed to initialize MCP server: {e}", "yellow"))
                print("Continuing without MCP server...")

        self.prompt = self.raw_prompt.format(
            username=self._user_name,
            user_id=self._user_id,
            memories="\n".join(self.memory),
        )

        if self.sent_prompt is False:
            res = self.chat.prompt(self.prompt, verbose=True)
            print("Chat result:", res)
            self.sent_prompt = True
        else:
            print(" -> We have already sent the prompt.")

        # This is from https://builtin.com/software-engineering-perspectives/discord-bot-python
        # LOOPS THROUGH ALL THE GUILD / SERVERS THAT THE BOT IS ASSOCIATED WITH.
        for guild in self.client.guilds:
            # PRINT THE SERVER'S ID AND NAME.
            print(f"Joining Server {guild.id} (name: {guild.name})")

            # INCREMENTS THE GUILD COUNTER.
            guild_count = guild_count + 1

            for channel in guild.text_channels:
                if channel.name == self.home_channel:
                    print(f"Adding home channel {channel} to the allowed channels.")
                    self.allowed_channels.add_home(channel)
                    break

        print(self.allowed_channels)

        # PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
        print("This bot is in " + str(guild_count) + " guild(s).")

        print("Starting the message processing queue.")
        self.process_queue.start()

        # Start reminder manager (background task checks every 10 seconds)
        self.reminder_manager.start()
        print(
            "Reminder system started - background task checking every 10 seconds for due reminders."
        )

        # Start scheduler (background task checks every 30 seconds)
        self.scheduler.start()
        print(
            "Scheduler system started - background task checking every 30 seconds for due scheduled tasks."
        )

        # Setup shutdown handler
        self._setup_shutdown_handler()

        print("Loaded conversation history:", len(self.chat))

    def _setup_shutdown_handler(self):
        """Setup signal handlers for graceful shutdown."""
        if self._shutdown_handler_setup:
            return

        self._shutting_down = False

        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            if self._shutting_down:
                # Already shutting down, force exit immediately
                print("\nForce exiting...")
                os._exit(1)

            self._shutting_down = True
            print("\nShutdown signal received.")

            # Stop background tasks immediately
            try:
                self.reminder_manager.stop()
                self.scheduler.stop()
            except Exception:
                pass

            # Signal handler runs in signal thread - don't try to send messages here
            # Raise KeyboardInterrupt in main thread to trigger graceful shutdown in async context
            import _thread

            _thread.interrupt_main()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._shutdown_handler_setup = True

    async def _send_goodbye_messages(self):
        """Send goodbye messages to all active channels."""
        try:
            if not self.client or not self.client.is_ready() or self.client.is_closed():
                return

            goodbye_message = "Goodbye! 👋"
            sent_count = 0

            # Send to all channels the bot is in
            for guild in self.client.guilds:
                for channel in guild.text_channels:
                    if channel in self.allowed_channels:
                        try:
                            # Check if client is still connected and channel is accessible
                            if self.client.is_closed():
                                break
                            if channel.permissions_for(guild.me).send_messages:
                                await channel.send(goodbye_message)
                                print(f"Sent goodbye to {channel.name}")
                                sent_count += 1
                        except (
                            discord.errors.HTTPException,
                            discord.errors.ConnectionClosed,
                        ) as e:
                            # Session closed or connection error - stop trying
                            print(f"Connection closed, stopping goodbye messages: {e}")
                            break
                        except Exception as e:
                            print(f"Error sending goodbye to {channel.name}: {e}")

            if sent_count > 0:
                print(f"Sent goodbye messages to {sent_count} channel(s)")
        except Exception as e:
            print(f"Error in goodbye handler: {e}")

    def _start_mcp_server(self):
        """Start the MCP server in a background task."""
        if not self.mcp_server:
            return

        # Note: MCP server via stdio typically runs as a subprocess invoked by MCP clients.
        # Running it alongside Discord bot in the same process is complex because both
        # need to handle stdin/stdout. For now, we'll just initialize it and log a message.
        # The --mcp-only mode is the recommended way to run Friend as an MCP server.
        print(
            colored(
                "MCP server initialized. Note: Running MCP alongside Discord bot "
                "is experimental. Use --mcp-only for standalone MCP server mode.",
                "yellow",
            )
        )
        if len(self.chat) == 0:
            print(" -> we will resend the prompt at the appropriate time.")
            print()
            print("Prompt:")
            print(self.prompt)
            print()

    def _extract_plain_text_from_llm_response(self, response: str) -> str:
        """
        Extract plain text from LLM response, removing tags and formatting.

        Args:
            response: Raw LLM response that may contain tags

        Returns:
            Clean plain text message
        """
        import re

        formatted_message = response.strip()
        # Remove <think>...</think> tags
        formatted_message = re.sub(
            r"<think>.*?</think>", "", formatted_message, flags=re.DOTALL
        )
        # Extract from <say> tags if present, otherwise use as-is
        say_match = re.search(r"<say>(.*?)</say>", formatted_message, re.DOTALL)
        if say_match:
            formatted_message = say_match.group(1).strip()
        # Remove any remaining tags
        formatted_message = re.sub(r"<[^>]+>", "", formatted_message)
        # Clean up whitespace
        formatted_message = " ".join(formatted_message.split())

        return formatted_message

    async def _handle_say_action(self, task: Task, content: str):
        """Handle the 'say' action - send message to channel.

        Args:
            task: The task containing channel and context information.
            content: The message content to send.
        """
        # Split content into <2000 character chunks
        while len(content) > 0:
            await task.channel.send(content[:2000])
            content = content[2000:]

    async def _handle_imagine_action(self, task: Task, content: str):
        """Handle the 'imagine' action - generate and send an image.

        Args:
            task: The task containing channel and context information.
            content: The image generation prompt.
        """
        await task.channel.send("*Imagining: " + content + "...*")
        time.sleep(0.1)  # Wait for message to be sent
        print("Generating image for prompt:", content)
        with self._chat_lock:
            image = self.image_service.generate_image(content)
        image.save("generated_image.png")

        # Send an image using the message service
        print(" - Sending content:", image)
        if self.message_service:
            await self.message_service.send_image(
                image,
                channel_id=str(task.channel.id),
                caption=None,
            )
        else:
            # Fallback to direct Discord send if service not initialized
            byte_arr = io.BytesIO()
            image.save(byte_arr, format="PNG")
            byte_arr.seek(0)
            file = discord.File(byte_arr, filename="image.png")
            await task.channel.send(file=file)

    async def _handle_remember_action(self, task: Task, content: str):
        """Handle the 'remember' action - add to memory.

        Args:
            task: The task containing channel and context information.
            content: The content to remember.
        """
        print("Remembering:", content)
        # Add this to memory
        self.memory.append(content)

        # Save memory to file
        with open("memory.txt", "w") as file:
            for line in self.memory:
                file.write(line + "\n")

        await task.channel.send("*Remembering: " + content + "*")

    async def _handle_forget_action(self, task: Task, content: str):
        """Handle the 'forget' action - remove from memory.

        Args:
            task: The task containing channel and context information.
            content: The content to forget.
        """
        print("Forgetting:", content)

        # Remove this from memory
        try:
            self.memory.remove(content)
        except ValueError:
            print(colored(" -> Could not find this in memory: ", content, "red"))

        # Save memory to file
        with open("memory.txt", "w") as file:
            for line in self.memory:
                file.write(line + "\n")

        await task.channel.send("*Forgetting: " + content + "*")

    async def _handle_weather_action(self, task: Task, content: str):
        """Handle the 'weather' action - get and send weather information.

        Args:
            task: The task containing channel and context information.
            content: The city/location to get weather for.
        """
        print("Getting weather for:", content)
        if not self.weather_api_key:
            await task.channel.send("*Sorry, weather API key is not configured.*")
        elif not self._weather_api_key_valid:
            await task.channel.send(
                "*Sorry, weather API key is invalid. Please check your configuration.*"
            )
        else:
            try:
                # Parse city from content (could be "London,UK" or just "London")
                city = content.strip()
                weather = get_current_weather(self.weather_api_key, city)
                weather_message = format_weather_message(weather)
                await task.channel.send(weather_message)
            except ValueError as e:
                # ValueError indicates API issues (invalid key, city not found, etc.)
                error_msg = f"*Error getting weather: {str(e)}*"
                print(colored(f"Weather error: {e}", "red"))
                await task.channel.send(error_msg)
            except Exception as e:
                # Other unexpected errors
                error_msg = f"*Unexpected error getting weather: {str(e)}*"
                print(colored(f"Weather error: {e}", "red"))
                await task.channel.send(error_msg)

    async def _handle_edit_image_action(self, task: Task, content: str):
        """Handle the 'edit_image' action - edit an image using Qwen Image Layered.

        Args:
            task: The task containing channel, attachments, and context information.
            content: The editing prompt/instructions.
        """
        # Edit an image using Qwen Image Layered
        if not isinstance(self.image_generator, QwenLayeredImageGenerator):
            await task.channel.send(
                "*Sorry, image editing is only available with the qwen-layered image generator.*"
            )
            return

        # Check if there are image attachments
        if not task.attachments or len(task.attachments) == 0:
            await task.channel.send(
                "*Sorry, no image was found to edit. Please send an image with your message.*"
            )
            return

        # Download the first image attachment
        attachment = task.attachments[0]
        try:
            await task.channel.send(f"*Editing image: {attachment.filename}...*")
            print(f"Downloading image: {attachment.url}")

            # Download image using Discord's attachment URL
            try:
                # Use discord.py's built-in attachment reading
                image_bytes = await attachment.read()
                input_image = Image.open(io.BytesIO(image_bytes))
                # Ensure RGBA for Qwen
                if input_image.mode != "RGBA":
                    input_image = input_image.convert("RGBA")
            except Exception as download_error:
                await task.channel.send(
                    f"*Error downloading image: {str(download_error)}*"
                )
                print(colored(f"Image download error: {download_error}", "red"))
                return

            # Edit the image with Qwen Image Layered
            edit_prompt = content if content else "enhance this image"
            print(f"Editing image with prompt: {edit_prompt}")

            with self._chat_lock:
                edited_image = self.image_generator.generate(
                    prompt=edit_prompt,
                    input_image=input_image,
                )

            edited_image.save("edited_image.png")

            # Send the edited image
            print(" - Sending edited image")
            if self.message_service:
                await self.message_service.send_image(
                    edited_image,
                    channel_id=str(task.channel.id),
                    caption=f"*Edited: {edit_prompt}*",
                )
            else:
                # Fallback to direct Discord send
                byte_arr = io.BytesIO()
                edited_image.save(byte_arr, format="PNG")
                byte_arr.seek(0)
                file = discord.File(byte_arr, filename="edited_image.png")
                await task.channel.send(f"*Edited: {edit_prompt}*", file=file)

        except Exception as e:
            error_msg = f"*Error editing image: {str(e)}*"
            print(colored(f"Image editing error: {e}", "red"))
            import traceback

            traceback.print_exc()
            await task.channel.send(error_msg)

    async def handle_task(self, task: Task):
        """Handle a task by processing the message and executing actions.

        This method processes incoming tasks by:
        1. Sending the message to the AI for processing
        2. Parsing the AI's response into action plans
        3. Routing each action to the appropriate handler

        Args:
            task: The task containing message, channel, and context information.
        """
        print()
        print("-" * 40)
        print("Handling task from channel:", task.channel.name)
        print("Handling task: message = \n", task.message)

        text = task.message
        try:
            if task.explicit:
                print("This task was explicitly triggered.")
                await task.channel.send(task.message)
                return
        except Exception as e:
            print(colored("Error in handling task: " + str(e), "red"))

        response = None
        # try:
        # Now actually prompt the AI
        with self._chat_lock:
            response = self.chat.prompt(
                text, verbose=True, assistant_history_prefix=""
            )  # f"{self._user_name} on #{channel_name}: ")
        action_plan = self.parser.parse(response)
        print()
        print("Action plan:", action_plan)
        for item in action_plan:
            # Handle both old format (action, content) and new format (action, content, attributes)
            if len(item) == 3:
                action, content, attributes = item
            else:
                action, content = item
                attributes = {}
            print(f"Action: {action}, Content: {content}, Attributes: {attributes}")
            # Route to appropriate action handler
            if action == "say":
                await self._handle_say_action(task, content)
            elif action == "imagine":
                await self._handle_imagine_action(task, content)
            elif action == "remember":
                await self._handle_remember_action(task, content)
            elif action == "forget":
                await self._handle_forget_action(task, content)
            elif action == "weather":
                await self._handle_weather_action(task, content)
            elif action == "edit_image":
                await self._handle_edit_image_action(task, content)
            elif action == "remind":
                await self._handle_remind_action(task, content, attributes)
            elif action == "schedule":
                await self._handle_schedule_action(task, content, attributes)
            elif action == "show_schedule":
                await self._handle_show_schedule_action(task, content)
            elif action == "help":
                await self._handle_help_action(task, content)

    def _parse_reminder_time(self, time_str: str, content: str) -> Optional[dict]:
        """
        Parse reminder time from attribute string.

        Args:
            time_str: Time string from attribute (e.g., "00:30:00" or "2025-12-20 15:00:00")
            content: Reminder message content

        Returns:
            Dictionary with reminder info or None if parsing fails
        """
        try:
            # Try parsing as relative time (HH:MM:SS)
            time_parts = time_str.split(":")
            if len(time_parts) == 3:
                hours, minutes, seconds = map(int, time_parts)
                time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                reminder_time = datetime.now() + time_delta
                return {
                    "time_delta": time_delta,
                    "reminder_time": reminder_time,
                    "message": content.strip() if content else "",
                    "users": [],
                }
            elif len(time_parts) == 2:
                # Try parsing as HH:MM (assume seconds = 0)
                hours, minutes = map(int, time_parts)
                time_delta = timedelta(hours=hours, minutes=minutes, seconds=0)
                reminder_time = datetime.now() + time_delta
                return {
                    "time_delta": time_delta,
                    "reminder_time": reminder_time,
                    "message": content.strip() if content else "",
                    "users": [],
                }
            else:
                # Try parsing as absolute time (YYYY-MM-DD HH:MM:SS or ISO format)
                # Handle both space and T separators
                time_str_normalized = time_str.replace(" ", "T")
                if "T" not in time_str_normalized and len(time_str_normalized) == 8:
                    # Might be just HH:MM:SS, treat as relative
                    time_parts = time_str.split(":")
                    if len(time_parts) == 3:
                        hours, minutes, seconds = map(int, time_parts)
                        time_delta = timedelta(
                            hours=hours, minutes=minutes, seconds=seconds
                        )
                        reminder_time = datetime.now() + time_delta
                        return {
                            "time_delta": time_delta,
                            "reminder_time": reminder_time,
                            "message": content.strip() if content else "",
                            "users": [],
                        }
                else:
                    reminder_time = datetime.fromisoformat(time_str_normalized)
                    return {
                        "time_delta": None,
                        "reminder_time": reminder_time,
                        "message": content.strip() if content else "",
                        "users": [],
                    }
        except (ValueError, TypeError):
            return None

    async def _handle_remind_action(self, task: Task, content: str, attributes: dict):
        """Handle the 'remind' action - create a reminder."""
        # Reminders are ONLY created when Friend explicitly uses <remind> action
        # Format: <remind time="HH:MM:SS">message</remind> or <remind time="YYYY-MM-DD HH:MM:SS">message</remind>
        # Friend must always specify the time attribute - we do NOT parse from user message

        # Check if time attribute is provided in the tag (REQUIRED)
        if not attributes.get("time"):
            await task.channel.send(
                '*Error: The <remind> tag requires a \'time\' attribute. Please use format like <remind time="00:30:00">message</remind> for relative time or <remind time="2025-12-20 15:00:00">message</remind> for absolute time.*'
            )
            return

        # Parse time from attribute
        time_str = attributes["time"]
        reminder_info_to_use = self._parse_reminder_time(time_str, content)

        if not reminder_info_to_use:
            print(f"Error parsing time attribute '{time_str}'")
            await task.channel.send(
                f"*Error: Could not parse time '{time_str}'. Please use format like '00:30:00' for 30 minutes or '2025-12-20 15:00:00' for absolute time.*"
            )
            return

        # Use AI's content as the reminder message
        reminder_message = content.strip() if content else ""

        # Determine reminder time
        if reminder_info_to_use.get("reminder_time"):
            reminder_time = reminder_info_to_use["reminder_time"]
            time_desc = reminder_time.strftime("%I:%M %p")
        elif reminder_info_to_use.get("time_delta"):
            reminder_time = datetime.now() + reminder_info_to_use["time_delta"]
            time_desc = str(reminder_info_to_use["time_delta"])
        else:
            await task.channel.send(
                "*I couldn't parse the reminder time. Please specify a time.*"
            )
            return

        # Get users to remind (from parsed info or default to current user)
        users_to_remind = reminder_info_to_use.get("users", [])
        if not users_to_remind:
            # Default to current user
            users_to_remind = [task.user_name]

        # Create reminders for each user
        reminders_created = []
        for user_name in users_to_remind:
            # Use current user's ID if it's "me" or the requester
            user_id_to_use = (
                task.user_id
                if user_name.lower() in ["me", task.user_name.lower()]
                else None
            )
            user_name_to_use = (
                task.user_name
                if user_name.lower() in ["me", task.user_name.lower()]
                else user_name
            )

            reminder = self.reminder_manager.add_reminder(
                channel_id=task.channel.id,
                channel_name=task.channel.name,
                user_id=user_id_to_use or task.user_id,
                user_name=user_name_to_use,
                reminder_time=reminder_time,
                message=reminder_message,
            )
            reminders_created.append(reminder)

        # Use the model to format the reminder message (async to avoid blocking)
        time_str = (
            time_desc
            if reminder_info_to_use.get("reminder_time")
            else str(reminder_info_to_use.get("time_delta", ""))
        )
        reminder_prompt = f"User {task.user_name} asked me to remind {'them' if len(users_to_remind) == 1 else f'{len(users_to_remind)} people'}: '{reminder_message}'. Format a friendly reminder message to send {'them' if len(users_to_remind) == 1 else 'them'} in {time_str}."

        # Format the reminder message in background to avoid blocking
        async def format_and_save_reminder():
            try:
                # Run the blocking chat.prompt() call in a thread executor to avoid blocking the event loop
                def _prompt_with_lock():
                    with self._chat_lock:
                        return self.chat.prompt(
                            reminder_prompt,
                            verbose=False,
                            assistant_history_prefix="",
                        )

                loop = asyncio.get_event_loop()
                formatted_response = await loop.run_in_executor(None, _prompt_with_lock)

                # Extract plain text from the formatted response
                formatted_message = self._extract_plain_text_from_llm_response(
                    formatted_response
                )

                # Use formatted message if we got something meaningful, otherwise use original
                if formatted_message and len(formatted_message) > 3:
                    for reminder in reminders_created:
                        reminder.message = formatted_message
                else:
                    for reminder in reminders_created:
                        reminder.message = reminder_message

                self.reminder_manager._save_reminders()
            except Exception as e:
                print(f"Error formatting reminder message: {e}")
                # Use the original message if formatting fails
                for reminder in reminders_created:
                    reminder.message = reminder_message
                self.reminder_manager._save_reminders()

        # Start formatting in background, don't wait
        asyncio.create_task(format_and_save_reminder())

        # Send confirmation immediately
        if reminder_info_to_use.get("reminder_time"):
            await task.channel.send(
                f"✓ Reminder set! I'll remind {', '.join(users_to_remind) if len(users_to_remind) > 1 else 'you'} at {time_desc}."
            )
        else:
            await task.channel.send(
                f"✓ Reminder set! I'll remind {', '.join(users_to_remind) if len(users_to_remind) > 1 else 'you'} in {time_desc}."
            )

    async def _handle_schedule_action(self, task: Task, content: str, attributes: dict):
        """Handle the 'schedule' action - create a scheduled task.

        Args:
            task: The task containing channel and context information.
            content: The message/content to schedule.
            attributes: Dictionary containing schedule attributes:
                - type: Schedule type ("daily", "hourly", "weekly", "interval")
                - value: Schedule value (e.g., "14:30", "5 minutes", "monday 14:30")
                - task_type: "post" or "dm" (optional, defaults to "post")
                - channel: Channel name for "post" tasks (optional, uses current channel if not specified)
        """
        # LLM should provide type and value attributes - we don't parse user messages
        schedule_type = attributes.get("type")
        schedule_value = attributes.get("value")
        task_type = attributes.get("task_type", "post")  # Default to "post"
        channel_name = attributes.get("channel")  # Optional channel name

        # Strip "#" prefix from channel name if present
        if channel_name and channel_name.startswith("#"):
            channel_name = channel_name[1:]

        # Extract plain text from content, removing any action tags
        # The content should be the message to schedule, not action tags
        schedule_message = content.strip() if content else ""
        # Remove any action tags that might have leaked in
        import re

        schedule_message = re.sub(r"<[^>]+>", "", schedule_message)
        schedule_message = " ".join(schedule_message.split()).strip()

        # Validate required attributes
        if not schedule_type:
            await task.channel.send(
                "*Error: The <schedule> tag requires a 'type' attribute. "
                'Please use format like <schedule type="interval" value="5 minutes">message</schedule> '
                'or <schedule type="daily" value="14:30" channel="general">message</schedule>.*'
            )
            return

        if not schedule_value:
            await task.channel.send(
                "*Error: The <schedule> tag requires a 'value' attribute. "
                'Please use format like <schedule type="interval" value="5 minutes">message</schedule> '
                'or <schedule type="daily" value="14:30">message</schedule>.*'
            )
            return

        try:
            # Create scheduled task
            if task_type == "post":
                # Find channel by name, or use current channel if not specified
                channel = None

                if channel_name:
                    # Try to find channel by name
                    print(f"Looking for channel: '{channel_name}' (after stripping #)")
                    for ch in self.client.get_all_channels():
                        if ch.name == channel_name:
                            channel = ch
                            print(f"Found channel: {ch.name} (ID: {ch.id})")
                            break
                    if not channel:
                        # List available channels for debugging
                        available_channels = [
                            ch.name
                            for ch in self.client.get_all_channels()
                            if isinstance(ch, discord.TextChannel)
                        ]
                        print(
                            f"Available channels: {available_channels[:10]}"
                        )  # Show first 10
                        await task.channel.send(
                            f"*Could not find channel '{channel_name}'. Available channels: {', '.join(available_channels[:5])}*"
                        )
                        return
                else:
                    # Use current channel
                    channel = task.channel
                    channel_name = channel.name

                scheduled_task = self.scheduler.add_task(
                    task_type="post",
                    message=schedule_message,
                    schedule_type=schedule_type,
                    schedule_value=schedule_value,
                    channel_id=channel.id,
                    channel_name=channel.name,
                )
                # Terminal output
                print(f"✓ Scheduled task created: ID={scheduled_task.task_id}")
                print("  Type: post")
                print(f"  Message: {schedule_message}")
                print(f"  Channel: #{channel_name} (ID: {channel.id})")
                print(f"  Schedule: {schedule_type} - {schedule_value}")
                print(f"  Next execution: {scheduled_task.next_execution}")

                # Format schedule description for user
                schedule_desc = f"{schedule_type}"
                if schedule_value:
                    schedule_desc += f" ({schedule_value})"

                await task.channel.send(
                    f"✓ Scheduled task created! I'll post '{schedule_message}' "
                    f"in #{channel_name} {schedule_desc}."
                )

            elif task_type == "dm":
                # Get user info - use task.user_id/user_name if available, otherwise from original message
                user_id_to_use = task.user_id
                user_name_to_use = task.user_name
                if not user_id_to_use and task.original_message:
                    user_id_to_use = task.original_message.author.id
                if not user_name_to_use and task.original_message:
                    user_name_to_use = task.original_message.author.display_name

                scheduled_task = self.scheduler.add_task(
                    task_type="dm",
                    message=schedule_message,
                    schedule_type=schedule_type,
                    schedule_value=schedule_value,
                    user_id=user_id_to_use,
                    user_name=user_name_to_use,
                )
                # Terminal output
                print(f"✓ Scheduled task created: ID={scheduled_task.task_id}")
                print("  Type: dm")
                print(f"  Message: {schedule_message}")
                print(f"  User: {user_name_to_use} (ID: {user_id_to_use})")
                print(f"  Schedule: {schedule_type} - {schedule_value}")
                print(f"  Next execution: {scheduled_task.next_execution}")

                # Format schedule description for user
                schedule_desc = f"{schedule_type}"
                if schedule_value:
                    schedule_desc += f" ({schedule_value})"

                await task.channel.send(
                    f"✓ Scheduled DM created! I'll DM you '{schedule_message}' "
                    f"{schedule_desc}."
                )
            else:
                await task.channel.send(
                    f"*Error: Invalid task_type '{task_type}'. Must be 'post' or 'dm'.*"
                )
                return
        except Exception as e:
            error_msg = f"*Error creating schedule: {str(e)}*"
            print(colored(f"Schedule error: {e}", "red"))
            import traceback

            traceback.print_exc()
            await task.channel.send(error_msg)

    async def _handle_show_schedule_action(self, task: Task, content: str):
        """Handle the 'show_schedule' action - display all scheduled tasks."""
        all_tasks = self.scheduler.get_all_tasks()

        if not all_tasks:
            await task.channel.send("*No scheduled tasks found.*")
            return

        # Terminal output
        print(f"📅 Showing {len(all_tasks)} scheduled task(s):")

        # Format tasks for display
        task_lines = []
        for task_item in all_tasks:
            if task_item.enabled:
                status = "✓ Enabled"
            else:
                status = "✗ Disabled"

            if task_item.task_type == "post":
                location = (
                    f"#{task_item.channel_name}"
                    if task_item.channel_name
                    else "unknown channel"
                )
            else:
                location = (
                    f"DM to {task_item.user_name}"
                    if task_item.user_name
                    else "unknown user"
                )

            schedule_desc = f"{task_item.schedule_type}"
            if task_item.schedule_value:
                schedule_desc += f" ({task_item.schedule_value})"

            next_exec = "N/A"
            if task_item.next_execution:
                try:
                    next_dt = datetime.fromisoformat(task_item.next_execution)
                    next_exec = next_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    next_exec = task_item.next_execution

            task_info = (
                f"**ID: {task_item.task_id}** - {status}\n"
                f"  Type: {task_item.task_type} → {location}\n"
                f"  Message: {task_item.message[:50]}{'...' if len(task_item.message) > 50 else ''}\n"
                f"  Schedule: {schedule_desc}\n"
                f"  Next execution: {next_exec}"
            )
            task_lines.append(task_info)

            # Terminal output
            print(f"  Task {task_item.task_id}: {task_item.task_type} → {location}")
            print(f"    Message: {task_item.message}")
            print(f"    Schedule: {schedule_desc}")
            print(f"    Next execution: {next_exec}")
            print(f"    Status: {status}")

        # Send to channel (split into chunks if too long)
        response = (
            f"**📅 Scheduled Tasks ({len(all_tasks)} total):**\n\n"
            + "\n\n".join(task_lines)
        )

        # Split into chunks if too long (Discord limit is 2000 chars)
        while len(response) > 0:
            chunk = response[:1900]
            # Try to break at a task boundary
            last_newline = chunk.rfind("\n\n")
            if last_newline > 1000:  # Only break if we have a reasonable chunk
                chunk = response[:last_newline]
            await task.channel.send(chunk)
            response = response[len(chunk) :].lstrip()

    async def _handle_help_action(self, task: Task, content: str):
        """Handle the 'help' action - retrieve documentation.

        Args:
            task: The task containing channel and context information.
            content: Optional topic name. If empty, lists all available topics.
        """
        print("Help requested:", content)
        if not content or content.strip() == "":
            # List all available docs
            help_text = list_available_docs()
            help_text += "\n\n**Usage:** Ask for help with a specific topic, e.g., `help discord` or `help models`"
            await task.channel.send(help_text)
        else:
            # Get specific doc
            doc_name = content.strip().lower()
            available_docs = get_available_docs()

            if doc_name in available_docs:
                doc_summary = format_doc_summary(doc_name, max_length=1800)
                if doc_summary:
                    # Split into chunks if too long
                    while len(doc_summary) > 0:
                        chunk = doc_summary[:1800]
                        await task.channel.send(chunk)
                        doc_summary = doc_summary[1800:]
                else:
                    await task.channel.send(
                        f"*Error loading documentation for: {doc_name}*"
                    )
            else:
                error_msg = (
                    f"*Documentation topic '{doc_name}' not found.*\n\n"
                    f"Available topics: {', '.join(sorted(available_docs.keys()))}"
                )
                await task.channel.send(error_msg)
        # except Exception as e:
        #    print(colored("Error in prompting the AI: " + str(e), "red"))
        #    print(" ->     Text:", text)
        #    print(" -> Response:", response)

    async def _execute_reminder(self, reminder: Reminder):
        """
        Execute a reminder by sending a message to the appropriate channel/user.
        Uses LLM to generate contextual message based on current time and reminder context.

        Args:
            reminder: The Reminder object to execute
        """
        try:
            # Generate contextual message using LLM
            current_time = datetime.now()
            time_str = current_time.strftime("%I:%M %p on %B %d, %Y")

            reminder_prompt = f"""Time: {time_str}
According to schedule, you should remind {reminder.user_name} about: "{reminder.message}"

Generate a friendly, contextual reminder message. Be creative and interesting - you can share a fun fact, make a joke, or add context based on the time of day. Keep it concise (1-2 sentences max). Return ONLY the message text, no tags or formatting.

Reminder message:"""

            try:
                # Run the blocking chat.prompt() call in a thread executor to avoid blocking the event loop
                def _prompt_with_lock():
                    with self._chat_lock:
                        return self.chat.prompt(
                            reminder_prompt,
                            verbose=False,
                            assistant_history_prefix="",
                        )

                loop = asyncio.get_event_loop()
                generated_message = await loop.run_in_executor(None, _prompt_with_lock)

                # Extract plain text from response
                generated_message = self._extract_plain_text_from_llm_response(
                    generated_message
                )

                # Use generated message if meaningful, otherwise fall back to original
                if generated_message and len(generated_message) > 3:
                    final_message = generated_message
                else:
                    final_message = reminder.message
            except Exception as e:
                print(f"Error generating reminder message with LLM: {e}")
                final_message = reminder.message

            # Format the reminder message
            if reminder.channel_id:
                # Send to channel
                channel = self.client.get_channel(reminder.channel_id)
                if channel:
                    # Check if we can send messages to this channel
                    if channel.permissions_for(channel.guild.me).send_messages:
                        message = f"🔔 <@{reminder.user_id}> Reminder: {final_message}"
                        await channel.send(message)
                        print(
                            f"Reminder executed: Sent to channel {reminder.channel_name}"
                        )
                    else:
                        # Fallback to DM if we can't send to channel
                        user = self.client.get_user(reminder.user_id)
                        if user:
                            message = f"🔔 Reminder (from #{reminder.channel_name}): {final_message}"
                            await user.send(message)
                            print(
                                f"Reminder executed: Sent DM to {reminder.user_name} (couldn't send to channel)"
                            )
                        else:
                            print(
                                f"Warning: Could not send reminder to channel or user {reminder.user_id}"
                            )
                else:
                    print(
                        f"Warning: Could not find channel {reminder.channel_id} for reminder"
                    )
            else:
                # Send DM to user
                user = self.client.get_user(reminder.user_id)
                if user:
                    message = f"🔔 Reminder: {final_message}"
                    await user.send(message)
                    print(f"Reminder executed: Sent DM to {reminder.user_name}")
                else:
                    print(
                        f"Warning: Could not find user {reminder.user_id} for reminder"
                    )
        except discord.errors.Forbidden:
            # Permission denied - try DM as fallback
            try:
                user = self.client.get_user(reminder.user_id)
                if user:
                    channel_info = (
                        f" (from #{reminder.channel_name})"
                        if reminder.channel_name
                        else ""
                    )
                    message = f"🔔 Reminder{channel_info}: {final_message}"
                    await user.send(message)
                    print(
                        f"Reminder executed: Sent DM to {reminder.user_name} (permission denied for channel)"
                    )
            except Exception as e2:
                print(
                    colored(
                        f"Error executing reminder (fallback DM also failed): {e2}",
                        "red",
                    )
                )
        except Exception as e:
            print(colored(f"Error executing reminder: {e}", "red"))
            import traceback

            traceback.print_exc()

    async def _execute_scheduled_task(self, task: ScheduledTask):
        """
        Execute a scheduled task by sending a message to the appropriate channel/user.
        Uses LLM to generate contextual message based on current time and schedule context.

        Args:
            task: The ScheduledTask object to execute
        """
        try:
            # Generate contextual message using LLM
            current_time = datetime.now()
            time_str = current_time.strftime("%I:%M %p on %B %d, %Y")

            schedule_context = f"{task.schedule_type}"
            if task.schedule_value:
                schedule_context += f" at {task.schedule_value}"

            task_prompt = f"""Time: {time_str}
According to schedule ({schedule_context}), you should post: "{task.message}"

Generate a friendly, contextual message. Be creative and interesting - you can share a fun fact, make a joke, add context based on the time of day, or expand on the original message. Keep it concise (1-3 sentences max). Return ONLY the message text, no tags or formatting.

Message to post:"""

            try:
                # Run the blocking chat.prompt() call in a thread executor to avoid blocking the event loop
                def _prompt_with_lock():
                    with self._chat_lock:
                        return self.chat.prompt(
                            task_prompt,
                            verbose=False,
                            assistant_history_prefix="",
                        )

                loop = asyncio.get_event_loop()
                generated_message = await loop.run_in_executor(None, _prompt_with_lock)

                # Extract plain text from response
                generated_message = self._extract_plain_text_from_llm_response(
                    generated_message
                )

                # Use generated message if meaningful, otherwise fall back to original
                if generated_message and len(generated_message) > 3:
                    final_message = generated_message
                else:
                    final_message = task.message
            except Exception as e:
                print(f"Error generating scheduled task message with LLM: {e}")
                final_message = task.message

            if task.task_type == "post":
                # Post to channel
                if task.channel_id:
                    channel = self.client.get_channel(task.channel_id)
                    if channel:
                        # Check permissions
                        if channel.permissions_for(channel.guild.me).send_messages:
                            await channel.send(final_message)
                            print(
                                f"Scheduled task executed: Posted to channel {task.channel_name}"
                            )
                        else:
                            print(
                                f"Warning: No permission to send to channel {task.channel_name}"
                            )
                    else:
                        print(
                            f"Warning: Could not find channel {task.channel_id} for scheduled task"
                        )
                else:
                    print(
                        f"Warning: Scheduled post task {task.task_id} has no channel_id"
                    )

            elif task.task_type == "dm":
                # Send DM to user
                if task.user_id:
                    user = self.client.get_user(task.user_id)
                    if user:
                        await user.send(final_message)
                        print(f"Scheduled task executed: Sent DM to {task.user_name}")
                    else:
                        print(
                            f"Warning: Could not find user {task.user_id} for scheduled task"
                        )
                else:
                    print(f"Warning: Scheduled DM task {task.task_id} has no user_id")
        except Exception as e:
            print(colored(f"Error executing scheduled task: {e}", "red"))
            import traceback

            traceback.print_exc()

    def on_message(self, message: discord.Message, verbose: bool = False):
        """Event listener for whenever a new message is sent to a channel that this bot is in."""
        if verbose:
            # Printing some information to learn about what this actually does
            print(message)
            print("Content =", message.content)
            print("Content type =", type(message.content))
            print("Author name:", message.author.name)
            print("Author global name:", message.author.global_name)

        # This is your actual username
        # sender_name = message.author.name
        sender_name = message.author.display_name
        # Only necessary once we want multi-server Friends
        # global_name = message.author.global_name

        # Skip anything that's from this bot
        if message.author.id == self._user_id:
            return None

        # TODO: make this a command line parameter for which channel(s) he should be in
        channel_name = message.channel.name
        print("Channel name:", channel_name)
        channel_id = message.channel.id
        print("Channel ID:", channel_id)
        # datetime = message.created_at

        timestamp = message.created_at
        print(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # Check if this channel is in the whitelist
        if self.join_at_random and channel_id not in self.allowed_channels:
            # Random number generator - 1 in 1000 chance
            random_number = random.randint(1, 100)
            print("Random number:", random_number)
            if random_number < 2:
                # Add to whitelist
                self.allowed_channels.visit(
                    channel_id, timeout_s=self.attention_window_seconds
                )
                print(f" -> Added {channel_name} to whitelist")

        print(self.allowed_channels)
        if message.channel not in self.allowed_channels:
            print(" -> Not in allowed channels. Skipping.")
            return None

        # Check for image attachments
        image_attachments = []
        if message.attachments:
            for attachment in message.attachments:
                # Check if attachment is an image
                if attachment.content_type and attachment.content_type.startswith(
                    "image/"
                ):
                    image_attachments.append(attachment)
                    print(
                        f"Found image attachment: {attachment.filename} ({attachment.content_type})"
                    )

        # Check for reminder/schedule keywords in the message (for context only, not auto-parsing)
        # Reminders and schedules should only be created when Friend explicitly uses <remind> or <schedule> actions
        message_content = message.content
        has_reminder_keywords = False
        has_schedule_keywords = False
        if message_content:
            # Check if message contains reminder keywords (for context hint only)
            reminder_keywords = ["remind", "reminder", "remind me", "remind us"]
            has_reminder_keywords = any(
                keyword in message_content.lower() for keyword in reminder_keywords
            )

            # Check if message contains schedule keywords (for context hint only)
            schedule_keywords = [
                "schedule",
                "always post",
                "always dm",
                "every",
                "hourly",
                "daily",
                "weekly",
            ]
            has_schedule_keywords = any(
                keyword in message_content.lower() for keyword in schedule_keywords
            )

        # Construct the text to prompt the AI
        # Include note about images if present
        text = f"{sender_name} on #{channel_name}: " + message_content
        if image_attachments:
            text += f" [User sent {len(image_attachments)} image(s) that can be edited]"
        if has_reminder_keywords:
            # Hint to AI that user mentioned a reminder, but don't auto-parse
            text += " [User mentioned a reminder - use <remind> action if you want to create one]"
        if has_schedule_keywords:
            # Hint to AI that user mentioned scheduling, but don't auto-parse
            text += " [User mentioned scheduling - use <schedule> action with type and value attributes if you want to create one]"

        self.push_task(
            channel=message.channel,
            message=text,
            attachments=image_attachments,
            user_id=message.author.id,
            user_name=sender_name,
            original_message=message,  # Store original Discord message object
        )

        print("Current task queue: ", self.task_queue.qsize())
        print("Current history length:", len(self.chat))
        # print(" -> Response:", response)
        return None

    def run(self):
        """Override run to handle shutdown gracefully with goodbye messages."""
        self.running = True

        async def _main():
            try:
                async with self.client as bot:
                    await bot.start(self.token)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Sending goodbye messages...")
                # Send goodbye messages before closing (client is still open here in async context)
                try:
                    if (
                        self.client
                        and self.client.is_ready()
                        and not self.client.is_closed()
                    ):
                        # Use say_goodbye() which is simpler and more reliable
                        await asyncio.wait_for(self.say_goodbye(), timeout=3.0)
                except asyncio.TimeoutError:
                    print("Timeout sending goodbye messages")
                except (
                    discord.errors.ConnectionClosed,
                    discord.errors.HTTPException,
                ) as e:
                    print(f"Connection already closed, skipping goodbye messages: {e}")
                except Exception as e:
                    print(f"Error sending goodbye messages: {e}")
                # Stop background tasks
                try:
                    self.reminder_manager.stop()
                    self.scheduler.stop()
                except Exception:
                    pass
            except Exception as e:
                print(f"Error in bot main loop: {e}")
                import traceback

                traceback.print_exc()
            finally:
                self.running = False

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
        except Exception as e:
            print(f"Error in bot run: {e}")
            self.running = False


@click.command()
@click.option("--token", default=None, help="The token for the discord bot.")
@click.option("--backend", default="gemma", help="The backend to use for the chat.")
@click.option(
    "--max-history-length", default=25, help="The maximum length of the chat history."
)
@click.option("--prompt", default="prompt.txt", help="The filename for the prompt.")
@click.option(
    "--image-generator",
    default="diffuser",
    help="The image generator to use.",
    type=click.Choice(["diffuser", "flux", "qwen-layered"], case_sensitive=False),
)
@click.option(
    "--weather-api-key",
    default=None,
    envvar="OPENWEATHER_API_KEY",
    help="OpenWeatherMap API key for weather functionality. Get one at https://openweathermap.org/api. Can also be set via OPENWEATHER_API_KEY environment variable.",
)
@click.option(
    "--enable-mcp",
    is_flag=True,
    default=False,
    help="Enable MCP server alongside Discord bot (experimental).",
)
@click.option(
    "--mcp-only",
    is_flag=True,
    default=False,
    help="Run as MCP server only (no Discord bot). Exposes Friend capabilities via MCP protocol.",
)
def main(
    token,
    backend,
    max_history_length,
    prompt,
    image_generator,
    weather_api_key,
    enable_mcp,
    mcp_only,
):
    if mcp_only:
        # Run as MCP server only
        print("Starting Friend as MCP server only...")

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
            image_gen = image_generator

        # Create services
        image_service = VirgilImageService(image_gen)

        # Create MCP server (without message service since no Discord)
        try:
            mcp_server = VirgilMCPServer(
                image_service=image_service,
                message_service=None,  # No Discord messaging in MCP-only mode
            )
            print("MCP server initialized. Starting...")
            asyncio.run(mcp_server.run())
        except Exception as e:
            print(colored(f"Failed to start MCP server: {e}", "red"))
            raise
    else:
        # Run as Discord bot (with optional MCP)
        bot = Friend(
            token=token,
            backend=backend,
            max_history_length=max_history_length,
            prompt_filename=prompt,
            image_generator=image_generator,
            weather_api_key=weather_api_key,
            enable_mcp=enable_mcp,
        )

        @bot.client.command(name="summon", help="Summon the bot to a channel.")
        async def summon(ctx):
            """Summon the bot to a channel."""
            print("Summoning the bot.")
            print(" -> Channel name:", ctx.channel.name)
            print(" -> Channel ID:", ctx.channel.id)
            bot.allowed_channels.visit(ctx.channel)
            await ctx.send("Hello! I am here to help you.")

        try:
            bot.run()
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            # Stop background tasks
            try:
                bot.reminder_manager.stop()
                bot.scheduler.stop()
            except Exception:
                pass
            # Send goodbye messages
            try:
                if bot.client and bot.client.is_ready():
                    # Use a timeout to prevent hanging
                    try:
                        asyncio.run(
                            asyncio.wait_for(bot._send_goodbye_messages(), timeout=2.0)
                        )
                    except asyncio.TimeoutError:
                        print("Timeout sending goodbye messages")
            except Exception as e:
                print(f"Error sending goodbye: {e}")


if __name__ == "__main__":
    main()
