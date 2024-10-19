# # (c) 2024 by Chris Paxton
# Useful reference: https://builtin.com/software-engineering-perspectives/discord-bot-python

from typing import Optional, List
import os
import time
import timeit
import discord
from discord.ext import commands, tasks
import asyncio
import queue

import threading

def read_discord_token_from_env():
    """Helpful tool to get a discord token from the command line, e.g. for a bot."""
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    TOKEN = os.getenv("DISCORD_TOKEN")
    return TOKEN


class Task:
    """Holds the fields for: message, channel, and content."""
    def __init__(self, message, channel, content):
        self.message = message
        self.channel = channel
        self.content = content


class DiscordBot:
    def __init__(self, token: Optional[str] = None):
        """Create the bot, using an authorization token from Discord."""
        # Create intents
        intents = discord.Intents.default()
        intents.message_content = True

        # Create an instance of a Client
        # self.client = discord.Client(intents=intents)
        self.client = commands.Bot(command_prefix="!", intents=intents)
        self._setup_hooks(self.client)

        # Save the token
        if token is None:
            token = read_discord_token_from_env()
        self.token = token

        self.running = True
        self.task_queue = queue.Queue()
        self.introduced_channels = set()

        # Whitelist of channels we can and will post in
        self.whitelist: Dict[str, float] = {}

        # Create a thread and lock for the message queue
        self.queue_lock = threading.Lock()
        # self.queue_thread = threading.Thread(target=self.process_queue)

    def add_to_whitelist(self, channel_id: str, current_time: Optional[float] = None):
        """Add a channel to the whitelist."""
        if current_time is None:
            current_time = timeit.default_timer()
        self.whitelist[channel_id] = current_time

    def remove_from_whitelist(self, channel_id: str):
        """Remove a channel from the whitelist."""
        del self.whitelist[channel_id]

    def channel_is_valid(self, channel, current_time: Optional[float] = None, threshold: float = 60) -> bool:
        """Check if a channel is valid to post in."""
        if channel.id in self.whitelist or channel.name in self.whitelist:
            print(" -> Channel is in the whitelist")
            last_time = self.whitelist[channel.id] if channel.id in self.whitelist else self.whitelist[channel.name]
            if current_time is None:
                current_time = timeit.default_timer()
            return current_time - last_time < threshold
        return False

    def push_task(self, channel, message: Optional[str] = None, content: Optional[str] = None):
        """Add a message to the queue to send."""
        self.task_queue.put(Task(message, channel, content))

    async def handle_task(self, task: Task):
        """Handle a task by sending the message to the channel. This will make the necessary calls in its thread to the different child functions that send messages, for example."""
        if task.message is not None:
            await task.channel.send(task.message)

    @tasks.loop(seconds=1)
    async def process_queue(self):
        """Process the queue of messages to send."""

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        while self.running:
            # Loop over all channels we have not yet started
            # Add a message for each one 
            for channel in self.client.get_all_channels():
                if channel.type == discord.ChannelType.text:
                    if channel not in self.introduced_channels:
                        introduced_channels.add(channel)
                        print("Check if channel is valid: ", channel.id, channel.name)
                        if self.channel_is_valid(channel):
                            print(f"Introducing myself to channel {channel.name}")
                            self.push_task(channel, message=self.greeting(), content=None)

            try:
                task = self.task_queue.get_nowait()
                await self.handle_task(task)
            except queue.Empty:
                await asyncio.sleep(1)  # Wait a bit before checking again

    def greeting(self) -> str:
        """Return a greeting message."""
        return "Hello everyone!"

    def _setup_hooks(self, client):
        """Prepare the various hooks to use this Bot object's methods."""

        @client.event
        async def on_ready():
            return self.on_ready()

        @client.event
        async def on_message(message):
            response = self.on_message(message)
            if response is not None:
                await message.channel.send(response)

    def on_ready(self):
        """Event listener called when the bot has switched from offline to online."""
        print(f"{self.client.user} has connected to Discord!")
        guild_count = 0

        print("Bot User name:", self.client.user.name)
        print("Bot Global name:", self.client.user.global_name)

        # This is from https://builtin.com/software-engineering-perspectives/discord-bot-python
        # LOOPS THROUGH ALL THE GUILD / SERVERS THAT THE BOT IS ASSOCIATED WITH.
        for guild in self.client.guilds:
            # PRINT THE SERVER'S ID AND NAME.
            print(f"- {guild.id} (name: {guild.name})")

            # INCREMENTS THE GUILD COUNTER.
            guild_count = guild_count + 1

        # PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
        print("This bot is in " + str(guild_count) + " guilds.")

    def on_message(self, message, verbose: bool = False):
        """Event listener for whenever a new message is sent to a channel that this bot is in."""
        if verbose:
            # Printing some information to learn about what this actually does
            print(message)
            print("Content =", message.content)
            print("Content type =", type(message.content))

        text = message.content
        if text.startswith("hello"):
            return "Hello!"

    def run(self):
        # Start the message thread to process the queue
        self.running = True
        self.client.run(self.token)

    def __del__(self):
        self.running = False


if __name__ == "__main__":
    bot = DiscordBot()
    bot.run()
