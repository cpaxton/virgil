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

        # Save the token
        self.token = token

    def run(self):
        self.client.run(token)


def read_discord_token_from_env():
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    TOKEN = os.getenv("DISCORD_TOKEN")
    return TOKEN


if __name__ == "__main__":
    token = read_discord_token_from_env()
    bot = DiscordBot(token)
    client = bot.client

    @client.event
    async def on_ready():
        print(f"{client.user} has connected to Discord!")

    @client.event
    async def on_message(message):
        print(message)
        print("Content =", message.content)
        print("Content type =", type(message.content))

        if message.author == client.user:
            print(message)
            return

        if message.content.startswith("$hello"):
            await message.channel.send("Hello!")

    bot.run()
