# # (c) 2024 by Chris Paxton

import os
import discord
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Create an instance of a Client
client = discord.Client()


@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("$hello"):
        await message.channel.send("Hello!")


# Run the bot with the token
client.run(TOKEN)
