# # (c) 2024 by Chris Paxton
# Usefuleight reference: https://builtin.com/software-engineering-perspectives/discord-bot-python

from typing import Optional, List
import os
import time
import timeit
import discord
from discord.ext import commands, tasks
import asyncio
from termcolor import colored
import queue
import io

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
    def __init__(self, message, channel, content, explicit: bool = False, t: float = None):
        self.message = message
        self.channel = channel
        self.content = content
        self.t = t  # This tracks the time the task was created

        # This tracks if we need to parse it or not
        self.explicit = explicit


class DiscordBot:
    def __init__(self, token: Optional[str] = None, timeout: float = 120):
        """Create the bot, using an authorization token from Discord."""
        # Create intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        # This is how long until we drop tasks
        self.timeout = timeout

        # Track if we have sent our startup message
        self._started = False

        # Store the user ID and name for the bot
        self._user_id = None
        self._user_name = None

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

    def push_task(self, channel, message: Optional[str] = None, content: Optional[str] = None, explicit: bool = False):
        """Add a message to the queue to send."""
        # print("Adding task to queue:", message, channel.name, content)
        self.task_queue.put(Task(message, channel, content, explicit=explicit, t=timeit.default_timer()))
        # print( "Queue length after push:", self.task_queue.qsize())

    async def handle_task(self, task: Task):
        """Handle a task by sending the message to the channel. This will make the necessary calls in its thread to the different child functions that send messages, for example."""
        print()
        print("Handling task: message = ", task.message, " channel = ", task.channel.name, " content = ", task.content)
        if task.message is not None:
            print(" - Sending message:", task.message)
            await task.channel.send(task.message)
        if task.content is not None:
            # Send an image
            print(" - Sending content:", task.content)
            # This should be a Discord file
            # Create a BytesIO object
            byte_arr = io.BytesIO()

            image = task.content
            # Save the image to the BytesIO object
            image.save(byte_arr, format="PNG")  # Save as PNG
            print( " - Image saved to byte array, format: ", image.format)

            # Move the cursor to the beginning of the BytesIO object
            byte_arr.seek(0)

            file = discord.File(byte_arr, filename="image.png")
            await task.channel.send(file=file)

    @tasks.loop(seconds=0.1)
    async def process_queue(self):
        """Process the queue of messages to send."""


        if not self._started:
            # Loop over all channels we have not yet started
            # Add a message for each one 
            for channel in self.client.get_all_channels():
                # print(" -", channel.id, channel.name, channel.type)
                if channel.type == discord.ChannelType.text:
                    # print(" -> Text channel")
                    if channel not in self.introduced_channels:
                        # print(" -> Not introduced yet")
                        self.introduced_channels.add(channel)
                        # print("Check if channel is valid: ", channel.id, channel.name)
                        if self.channel_is_valid(channel):
                            print(f"Introducing myself to channel {channel.name}")
                            try:
                                self.push_task(channel, message=self.greeting(), content=None, explicit=True)
                            except Exception as e:
                                print(colored("Error in introducing myself: " + str(e), "red"))
            self._started = True

        # Print queue length
        # print("Queue length:", self.task_queue.qsize())
        try:
            task = self.task_queue.get_nowait()
            if task.t + self.timeout < timeit.default_timer():
                print("Dropping task due to timeout: ", task.message, task.channel.name)
                return
            print("Handling task from queue:", task)
            await self.handle_task(task)
        except queue.Empty:
            await asyncio.sleep(0.1)  # Wait a bit before checking again
        except Exception as e:
            print(colored( "Error in processing queue: " + str(e), "red"))

    def greeting(self) -> str:
        """Return a greeting message."""
        return "Hello everyone!"

    def _setup_hooks(self, client):
        """Prepare the various hooks to use this Bot object's methods."""

        @client.event
        async def on_ready():
            self._started = False
            self._user_name = self.client.user.name
            self._user_id = self.client.user.id
            return self.on_ready()

        @client.event
        async def on_message(message):
            # This line is important to allow commands to work
            # await bot.process_commands(message)
    
            # Check if the bot was mentioned
            # print()
            # print("Mentions:", message.mentions)
            idx1 = message.content.find("@" + self._user_name)
            idx2 = message.content.find("<@" + str(self._user_id) + ">")
            if idx1 >= 0 or idx2 >= 0:
                print(" ->", self._user_name, " was mentioned in channel", message.channel.name, "with content:", message.content)
                if not self.channel_is_valid(message.channel):
                    # Add it to the whitelist since we were mentioned
                    self.add_to_whitelist(message.channel.id)

            print("Message content:", message.content)
            response = self.on_message(message)

            if response is not None:
                print('Sending response:', response)
                await message.channel.send(response)
                print("Done")

            print("-------------")

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

        print("Starting the message processing queue.")
        self.process_queue.start()


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

        async def _main():
            async with self.client as bot:
                await bot.start(self.token)

        # self.client.start(self.token)
        asyncio.run(_main())

    def __del__(self):
        self.running = False


if __name__ == "__main__":
    bot = DiscordBot()
    bot.run()
