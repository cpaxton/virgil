# # (c) 2024 by Chris Paxton

import os
import discord


class DiscordBot:
    def __init__(self, token):
        # Create intents
        intents = discord.Intents.default()
        intents.message_content = True

        # Create an instance of a Client
        self.client = discord.Client(intents=intents)

        self._setup_hooks(self.client)

        # Save the token
        self.token = token

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
            return "Hello"

    def run(self):
        self.client.run(token)


def read_discord_token_from_env():
    """Helpful tool to get a discord token from the command line, e.g. for a bot."""
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    TOKEN = os.getenv("DISCORD_TOKEN")
    return TOKEN


if __name__ == "__main__":
    token = read_discord_token_from_env()
    bot = DiscordBot(token)
    bot.run()
