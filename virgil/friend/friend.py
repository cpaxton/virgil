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
import sys
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
import aiohttp
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
from virgil.friend.reminder import ReminderManager, Reminder, parse_reminder_command
from virgil.friend.scheduler import Scheduler, ScheduledTask, parse_schedule_command
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
from datetime import datetime


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
    """Friend is a simple discord bot, which chats with you if you are on its server. Be patient with it, it's very stupid."""

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
            image_generator (Optional[DiffuserImageGenerator]): The image generator to use. Defaults to None.
            join_at_random (bool): Whether to join channels at random. Defaults to False.
            max_history_length (int): The maximum length of the chat history. Defaults to 25.
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

    def on_ready(self):
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

        # Start reminder manager
        self.reminder_manager.start()
        print("Reminder system started.")

        # Start scheduler
        self.scheduler.start()
        print("Scheduler system started.")

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
            print("\nShutdown signal received. Sending goodbye messages...")

            # Stop background tasks
            try:
                self.reminder_manager.stop()
                self.scheduler.stop()
            except Exception:
                pass

            # Try to send goodbye messages before closing
            try:
                if self.client and self.client.is_ready():
                    loop = None
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        pass

                    if loop and loop.is_running():
                        # Schedule goodbye messages and wait briefly for them to send
                        future = asyncio.run_coroutine_threadsafe(
                            asyncio.wait_for(
                                self._send_goodbye_messages(), timeout=2.0
                            ),
                            loop,
                        )
                        # Wait for messages to send (with timeout)
                        try:
                            future.result(timeout=2.5)
                        except Exception:
                            pass  # Timeout or error, continue with shutdown
            except Exception as e:
                print(f"Error in shutdown handler: {e}")

            # Exit
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._shutdown_handler_setup = True

    async def _send_goodbye_messages(self):
        """Send goodbye messages to all active channels."""
        try:
            if not self.client or not self.client.is_ready():
                return

            goodbye_message = "Goodbye! 👋"
            sent_count = 0

            # Send to all channels the bot is in
            for guild in self.client.guilds:
                for channel in guild.text_channels:
                    if channel in self.allowed_channels:
                        try:
                            # Check if channel is accessible before sending
                            if channel.permissions_for(guild.me).send_messages:
                                await channel.send(goodbye_message)
                                print(f"Sent goodbye to {channel.name}")
                                sent_count += 1
                        except discord.errors.HTTPException as e:
                            # Session closed or other HTTP error
                            print(f"Could not send goodbye to {channel.name}: {e}")
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

    async def handle_task(self, task: Task):
        """Handle a task by sending the message to the channel. This will make the necessary calls in its thread to the different child functions that send messages, for example."""
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
        for action, content in action_plan:
            print(f"Action: {action}, Content: {content}")  # Handle actions here
            if action == "say":
                # Split content into <2000 character chunks
                while len(content) > 0:
                    await task.channel.send(content[:2000])
                    content = content[2000:]
            elif action == "imagine":
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
            elif action == "remember":
                print("Remembering:", content)
                # Add this to memory
                self.memory.append(content)

                # Save memory to file
                with open("memory.txt", "w") as file:
                    for line in self.memory:
                        file.write(line + "\n")

                await task.channel.send("*Remembering: " + content + "*")
            elif action == "forget":
                print("Forgetting:", content)

                # Remove this from memory
                try:
                    self.memory.remove(content)
                except ValueError:
                    print(
                        colored(" -> Could not find this in memory: ", content, "red")
                    )

                # Save memory to file
                with open("memory.txt", "w") as file:
                    for line in self.memory:
                        file.write(line + "\n")

                await task.channel.send("*Forgetting: " + content + "*")
            elif action == "weather":
                print("Getting weather for:", content)
                if not self.weather_api_key:
                    await task.channel.send(
                        "*Sorry, weather API key is not configured.*"
                    )
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
            elif action == "edit_image":
                # Edit an image using Qwen Image Layered
                if not isinstance(self.image_generator, QwenLayeredImageGenerator):
                    await task.channel.send(
                        "*Sorry, image editing is only available with the qwen-layered image generator.*"
                    )
                    continue

                # Check if there are image attachments
                if not task.attachments or len(task.attachments) == 0:
                    await task.channel.send(
                        "*Sorry, no image was found to edit. Please send an image with your message.*"
                    )
                    continue

                # Download the first image attachment
                attachment = task.attachments[0]
                try:
                    await task.channel.send(
                        f"*Editing image: {attachment.filename}...*"
                    )
                    print(f"Downloading image: {attachment.url}")

                    # Download image using Discord's attachment URL
                    # Discord attachments can be accessed directly via their URL
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
                        continue

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
            elif action == "remind":
                # Handle reminder action from AI
                # Reminders are ONLY created when Friend explicitly uses <remind> action
                # Parse reminder from original user message or AI's content
                reminder_info_to_use = None

                # Try to parse from original user message first
                if task.message.content:
                    parsed_reminder = parse_reminder_command(task.message.content)

                    # If regex parsing fails, try LLM parsing
                    if not parsed_reminder:
                        from virgil.friend.reminder import (
                            parse_reminder_command_with_llm,
                        )

                        # Use the chat model to parse
                        def llm_parse_prompt(prompt_text):
                            with self._chat_lock:
                                return self.chat.prompt(
                                    prompt_text,
                                    verbose=False,
                                    assistant_history_prefix="",
                                )

                        parsed_reminder, parsing_instructions = (
                            parse_reminder_command_with_llm(
                                task.message.content, llm_parse_prompt
                            )
                        )
                        if parsing_instructions:
                            print(f"LLM parsing suggestions: {parsing_instructions}")

                    if parsed_reminder:
                        reminder_info_to_use = parsed_reminder.copy()
                        if reminder_info_to_use.get("time_delta"):
                            reminder_info_to_use["reminder_time"] = (
                                datetime.now() + reminder_info_to_use["time_delta"]
                            )
                        elif reminder_info_to_use.get("reminder_time"):
                            # Already has absolute time
                            pass
                        else:
                            reminder_info_to_use = None

                if reminder_info_to_use:
                    # Use the parsed reminder from user's message
                    # Use AI's content as the reminder message if provided, otherwise use original
                    reminder_message = (
                        content.strip() if content else reminder_info_to_use["message"]
                    )

                    # Determine reminder time
                    if reminder_info_to_use.get("reminder_time"):
                        reminder_time = reminder_info_to_use["reminder_time"]
                        time_desc = reminder_time.strftime("%I:%M %p")
                    elif reminder_info_to_use.get("time_delta"):
                        reminder_time = (
                            datetime.now() + reminder_info_to_use["time_delta"]
                        )
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
                    # For now, we'll create one reminder per user mentioned
                    # In the future, we could look up Discord users by name
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
                            with self._chat_lock:
                                formatted_response = self.chat.prompt(
                                    reminder_prompt,
                                    verbose=False,
                                    assistant_history_prefix="",
                                )

                            # Extract plain text from the formatted response
                            # Remove any action tags and thinking tags
                            formatted_message = formatted_response.strip()

                            # Remove <think>...</think> tags
                            import re

                            formatted_message = re.sub(
                                r"<think>.*?</think>",
                                "",
                                formatted_message,
                                flags=re.DOTALL,
                            )

                            # Extract content from <say> tags if present, otherwise use as-is
                            say_match = re.search(
                                r"<say>(.*?)</say>", formatted_message, re.DOTALL
                            )
                            if say_match:
                                formatted_message = say_match.group(1).strip()

                            # Remove any remaining tags
                            formatted_message = re.sub(
                                r"<[^>]+>", "", formatted_message
                            )

                            # Clean up whitespace
                            formatted_message = " ".join(formatted_message.split())

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
                elif content:
                    # Try to parse reminder from AI's content (fallback if no reminder_info)
                    parsed = parse_reminder_command(content)

                    # If regex parsing fails, try LLM parsing
                    if not parsed:
                        from virgil.friend.reminder import (
                            parse_reminder_command_with_llm,
                        )

                        # Use the chat model to parse
                        def llm_parse_prompt(prompt_text):
                            with self._chat_lock:
                                return self.chat.prompt(
                                    prompt_text,
                                    verbose=False,
                                    assistant_history_prefix="",
                                )

                        parsed, parsing_instructions = parse_reminder_command_with_llm(
                            content, llm_parse_prompt
                        )
                        if parsing_instructions:
                            print(f"LLM parsing suggestions: {parsing_instructions}")

                    if parsed:
                        # New format returns a dict
                        if parsed.get("reminder_time"):
                            reminder_time = parsed["reminder_time"]
                            time_desc = reminder_time.strftime("%I:%M %p")
                        elif parsed.get("time_delta"):
                            reminder_time = datetime.now() + parsed["time_delta"]
                            time_desc = str(parsed["time_delta"])
                        else:
                            await task.channel.send(
                                "*I couldn't parse the reminder time. Please specify a time.*"
                            )
                            return

                        reminder_msg = parsed["message"]
                        users_to_remind = parsed.get("users", [])
                        if not users_to_remind:
                            users_to_remind = [task.user_name]

                        # Create reminders for each user
                        for user_name in users_to_remind:
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

                            self.reminder_manager.add_reminder(
                                channel_id=task.channel.id,
                                channel_name=task.channel.name,
                                user_id=user_id_to_use or task.user_id,
                                user_name=user_name_to_use,
                                reminder_time=reminder_time,
                                message=reminder_msg,
                            )

                        if parsed.get("reminder_time"):
                            await task.channel.send(
                                f"✓ Reminder set! I'll remind {', '.join(users_to_remind) if len(users_to_remind) > 1 else 'you'} at {time_desc}."
                            )
                        else:
                            await task.channel.send(
                                f"✓ Reminder set! I'll remind {', '.join(users_to_remind) if len(users_to_remind) > 1 else 'you'} in {time_desc}."
                            )
                    else:
                        # Check if the message might be incomplete
                        if content and content.lower().strip().endswith(
                            ("to", "that", "about")
                        ):
                            await task.channel.send(
                                "*It looks like your reminder message might be incomplete. Please finish your reminder, like 'remind me in 30 seconds to check my email' or 'remind me in 30 seconds that I need to call mom'.*"
                            )
                        else:
                            await task.channel.send(
                                "*I couldn't parse the reminder. Please use format like 'remind me in 30 mins to do something' or 'remind me at 3pm to do something'.*"
                            )
                else:
                    await task.channel.send(
                        "*No reminder information provided. Please include timing information in your reminder request.*"
                    )
            elif action == "schedule":
                # Handle schedule action from AI
                # Parse schedule command from content or user's message
                schedule_info = None

                # Try to parse from AI's content first
                if content:
                    schedule_info = parse_schedule_command(content)

                # If not found, try parsing from original message
                if not schedule_info and task.message.content:
                    schedule_info = parse_schedule_command(task.message.content)

                if schedule_info:
                    try:
                        # Create scheduled task
                        if schedule_info["task_type"] == "post":
                            # Find channel by name
                            channel = None
                            channel_name = schedule_info.get("channel_name", "")
                            for ch in self.client.get_all_channels():
                                if ch.name == channel_name:
                                    channel = ch
                                    break

                            if channel:
                                self.scheduler.add_task(
                                    task_type="post",
                                    message=schedule_info["message"],
                                    schedule_type=schedule_info["schedule_type"],
                                    schedule_value=schedule_info["schedule_value"],
                                    channel_id=channel.id,
                                    channel_name=channel.name,
                                )
                                await task.channel.send(
                                    f"✓ Scheduled task created! I'll post '{schedule_info['message']}' "
                                    f"in #{channel_name} {schedule_info['schedule_type']} at {schedule_info['schedule_value']}."
                                )
                            else:
                                await task.channel.send(
                                    f"*Could not find channel '{channel_name}'. Please check the channel name.*"
                                )

                        elif schedule_info["task_type"] == "dm":
                            self.scheduler.add_task(
                                task_type="dm",
                                message=schedule_info["message"],
                                schedule_type=schedule_info["schedule_type"],
                                schedule_value=schedule_info["schedule_value"],
                                user_id=task.user_id or task.message.author.id,
                                user_name=task.user_name
                                or task.message.author.display_name,
                            )
                            await task.channel.send(
                                f"✓ Scheduled DM created! I'll DM you '{schedule_info['message']}' "
                                f"{schedule_info['schedule_type']} at {schedule_info['schedule_value']}."
                            )
                    except Exception as e:
                        error_msg = f"*Error creating schedule: {str(e)}*"
                        print(colored(f"Schedule error: {e}", "red"))
                        import traceback

                        traceback.print_exc()
                        await task.channel.send(error_msg)
                else:
                    await task.channel.send(
                        "*I couldn't parse the schedule command. "
                        "Try: 'always post X in Y channel at 14:30' or 'always DM me X at 14:30'*"
                    )
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
                with self._chat_lock:
                    generated_message = self.chat.prompt(
                        reminder_prompt,
                        verbose=False,
                        assistant_history_prefix="",
                    )

                # Extract plain text from response
                import re

                generated_message = generated_message.strip()
                # Remove <think> tags
                generated_message = re.sub(
                    r"<think>.*?</think>", "", generated_message, flags=re.DOTALL
                )
                # Extract from <say> tags if present
                say_match = re.search(r"<say>(.*?)</say>", generated_message, re.DOTALL)
                if say_match:
                    generated_message = say_match.group(1).strip()
                # Remove any remaining tags
                generated_message = re.sub(r"<[^>]+>", "", generated_message)
                generated_message = " ".join(generated_message.split())

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
                with self._chat_lock:
                    generated_message = self.chat.prompt(
                        task_prompt,
                        verbose=False,
                        assistant_history_prefix="",
                    )

                # Extract plain text from response
                import re

                generated_message = generated_message.strip()
                # Remove <think> tags
                generated_message = re.sub(
                    r"<think>.*?</think>", "", generated_message, flags=re.DOTALL
                )
                # Extract from <say> tags if present
                say_match = re.search(r"<say>(.*?)</say>", generated_message, re.DOTALL)
                if say_match:
                    generated_message = say_match.group(1).strip()
                # Remove any remaining tags
                generated_message = re.sub(r"<[^>]+>", "", generated_message)
                generated_message = " ".join(generated_message.split())

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

        timestamp = message.created_at.timestamp()
        print("Timestamp:", timestamp)

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
        # Reminders should only be created when Friend explicitly uses <remind> action
        schedule_info = None
        message_content = message.content
        has_reminder_keywords = False
        if message_content:
            # Check if message contains reminder keywords (for context hint only)
            reminder_keywords = ["remind", "reminder", "remind me", "remind us"]
            has_reminder_keywords = any(
                keyword in message_content.lower() for keyword in reminder_keywords
            )

            # Check for schedule commands (these are still auto-parsed for context)
            parsed_schedule = parse_schedule_command(message_content)
            if parsed_schedule:
                schedule_info = parsed_schedule

        # Construct the text to prompt the AI
        # Include note about images if present
        text = f"{sender_name} on #{channel_name}: " + message_content
        if image_attachments:
            text += f" [User sent {len(image_attachments)} image(s) that can be edited]"
        if has_reminder_keywords:
            # Hint to AI that user mentioned a reminder, but don't auto-parse
            text += " [User mentioned a reminder - use <remind> action if you want to create one]"
        if schedule_info:
            text += f" [User requested a scheduled task: {schedule_info['task_type']} '{schedule_info['message']}' {schedule_info['schedule_type']} at {schedule_info['schedule_value']}]"

        self.push_task(
            channel=message.channel,
            message=text,
            attachments=image_attachments,
            reminder_info=None,  # Don't auto-parse reminders - only create when AI uses <remind>
            user_id=message.author.id,
            user_name=sender_name,
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
                # Send goodbye messages before closing
                try:
                    if self.client and self.client.is_ready():
                        await asyncio.wait_for(
                            self._send_goodbye_messages(), timeout=3.0
                        )
                except asyncio.TimeoutError:
                    print("Timeout sending goodbye messages")
                except Exception as e:
                    print(f"Error sending goodbye messages: {e}")
                # Stop background tasks
                try:
                    self.reminder_manager.stop()
                    self.scheduler.stop()
                except Exception:
                    pass
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
