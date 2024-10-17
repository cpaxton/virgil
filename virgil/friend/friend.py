# # (c) 2024 by Chris Paxton

from virgil.io.discord_bot import DiscordBot
from virgil.backend import get_backend
from virgil.chat import ChatWrapper
from typing import Optional
import pkg_resources


def load_prompt():
    # Get the path to the quiz.html file
    file_path = pkg_resources.resource_filename("virgil.friend", "prompt.txt")

    # Read the contents of the file
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content


class Friend(DiscordBot):
    """Friend is a simple discord bot, which chats with you if you are on its server. Be patient with it, it's very stupid."""

    def __init__(self, token: Optional[str] = None, backend="gemma"):
        self.backend = get_backend(backend)
        self.chat = ChatWrapper(self.backend)
        super(Friend, self).__init__(token)

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
        print("This bot is in " + str(guild_count) + " guild(s).")
        print("Loaded conversation history:", len(self.chat))

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


if __name__ == "__main__":
    bot = Friend()
    bot.run()
