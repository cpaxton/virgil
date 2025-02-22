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
from virgil.io.discord_bot import DiscordBot, Task
from virgil.backend import get_backend
from virgil.chat import ChatWrapper
from typing import Optional
import timeit
import random
import threading
import time
from termcolor import colored
import io
import discord

# This only works on Ampere+ GPUs
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from virgil.friend.parser import ChatbotActionParser
from virgil.image.diffuser import DiffuserImageGenerator
from virgil.utils import load_prompt


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

    def __init__(self, token: Optional[str] = None, backend="gemma",
                 attention_window_seconds: float = 600.0, image_generator: str = "flux",
                 join_at_random: bool = False, max_history_length: int = 25,
                 prompt_filename: str = "prompt.txt", home_channel: str = "ask-a-robot") -> None:
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
        self.chat = ChatWrapper(self.backend, max_history_length=max_history_length, preserve=2)
        self.attention_window_seconds = attention_window_seconds
        self.raw_prompt = load_prompt_helper(prompt_filename)
        self.prompt = None
        self._user_name = None
        self._user_id = None
        self.sent_prompt = False
        self.join_at_random = join_at_random
        self.home_channel = home_channel
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
                self.image_generator = DiffuserImageGenerator(height=512, width=512, num_inference_steps=4, guidance_scale=0.0, model="turbo", xformers=False)
            elif image_generator.lower() == "flux":
                self.image_generator = FluxImageGenerator(height=512, width=512)
        else:
            self.image_generator = image_generator

        self.parser = ChatbotActionParser(self.chat)

        self._chat_lock = threading.Lock()  # Lock for chat access

    def on_ready(self):
        """Event listener called when the bot has switched from offline to online."""
        print(f"{self.client.user} has connected to Discord!")
        guild_count = 0

        print("Bot User name:", self.client.user.name)
        print("Bot Global name:", self.client.user.global_name)
        print("Bot User IDL", self.client.user.id)
        self._user_name = self.client.user.name
        self._user_id = self.client.user.id
        self.prompt = self.raw_prompt.format(username=self._user_name, user_id=self._user_id, memories="\n".join(self.memory))

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

        print("Loaded conversation history:", len(self.chat))
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
            response = self.chat.prompt(text, verbose=True, assistant_history_prefix="")  # f"{self._user_name} on #{channel_name}: ")
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
                    image = self.image_generator.generate(content)
                image.save("generated_image.png")

                # Send an image
                print(" - Sending content:", image)
                # This should be a Discord file
                # Create a BytesIO object
                byte_arr = io.BytesIO()

                # Save the image to the BytesIO object
                image.save(byte_arr, format="PNG")  # Save as PNG
                print(" - Image saved to byte array")

                # Move the cursor to the beginning of the BytesIO object
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
                    print(colored(" -> Could not find this in memory: ", content, "red"))

                # Save memory to file
                with open("memory.txt", "w") as file:
                    for line in self.memory:
                        file.write(line + "\n")

                await task.channel.send("*Forgetting: " + content + "*")
        # except Exception as e:
        #    print(colored("Error in prompting the AI: " + str(e), "red"))
        #    print(" ->     Text:", text)
        #    print(" -> Response:", response)

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
        if self.join_at_random and not channel_id in self.allowed_channels:
            # Random number generator - 1 in 1000 chance
            random_number = random.randint(1, 100)
            print("Random number:", random_number)
            if random_number < 2:
                # Add to whitelist
                self.allowed_channels.visit(channel_id, timeout_s=self.attention_window_seconds)
                print(f" -> Added {channel_name} to whitelist")

        print(self.allowed_channels)
        if not message.channel in self.allowed_channels:
            print(" -> Not in allowed channels. Skipping.")
            return None

        # Construct the text to prompt the AI
        text = f"{sender_name} on #{channel_name}: " + message.content
        self.push_task(channel=message.channel, message=text)

        print("Current task queue: ", self.task_queue.qsize())
        print("Current history length:", len(self.chat))
        # print(" -> Response:", response)
        return None


@click.command()
@click.option("--token", default=None, help="The token for the discord bot.")
@click.option("--backend", default="gemma", help="The backend to use for the chat.")
@click.option("--max-history-length", default=25, help="The maximum length of the chat history.")
@click.option("--prompt", default="prompt.txt", help="The filename for the prompt.")
def main(token, backend, max_history_length, prompt):
    bot = Friend(token=token, backend=backend, max_history_length=max_history_length, prompt_filename=prompt)
    client = bot.client

    @bot.client.command(name="summon", help="Summon the bot to a channel.")
    async def summon(ctx):
        """Summon the bot to a channel."""
        print("Summoning the bot.")
        print(" -> Channel name:", ctx.channel.name)
        print(" -> Channel ID:", ctx.channel.id)
        bot.allowed_channels.visit(ctx.channel)
        await ctx.send("Hello! I am here to help you.")

    bot.run()


if __name__ == "__main__":
    main()
