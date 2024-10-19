# # (c) 2024 by Chris Paxton

import click
from virgil.io.discord_bot import DiscordBot
from virgil.backend import get_backend
from virgil.chat import ChatWrapper
from typing import Optional
import pkg_resources
import timeit
import random


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
        self.prompt = load_prompt()
        self._user_name = None
        self._user_id = None
        self.whitelist: Dict[str, float] = {}
        super(Friend, self).__init__(token)

    def on_ready(self):
        """Event listener called when the bot has switched from offline to online."""
        print(f"{self.client.user} has connected to Discord!")
        guild_count = 0

        print("Bot User name:", self.client.user.name)
        print("Bot Global name:", self.client.user.global_name)
        print("Bot User IDL", self.client.user.id)
        self._user_name = self.client.user.name
        self._user_id = self.client.user.id

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
        if len(self.chat) == 0:
            print(" -> we will resend the prompt at the appropriate time.")
            print()
            print("Prompt:")
            print(self.prompt)
            print()

    def on_message(self, message, verbose: bool = False):
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
        if message.channel.name != "ask-a-robot":
            # Check name in whitelist
            t1 = timeit.default_timer()
            channel_name = message.channel.name
            ok = False

            print("Current whitelist channels: ", self.whitelist)

            if channel_name in self.whitelist:
                # Check if it was within last 10 mins
                t2 = self.whitelist[channel_name]
                if t1 - t2 < 60:
                    # This is ok
                    ok = True
            
            # Random number generator - 1 in 1000 chance
            random_number = random.randint(1, 100)
            print("Random number:", random_number)
            if random_number < 2:
                # Add to whitelist 
                self.whitelist[channel_name] = t1
                ok = True
                print(f" -> Added {channel_name} to whitelist")

            if not ok:
                return None

        if len(self.chat) == 0:
            text = self.prompt + f"\n{sender_name}: " + message.content
        else:
            text = f"{sender_name}: " + message.content

        # Now actually prompt the AI
        response = self.chat.prompt(text, verbose=True)
        return response

@click.command()
@click.option("--token", default=None, help="The token for the discord bot.")
@click.option("--backend", default="gemma", help="The backend to use for the chat.")
def main(token, backend):
    bot = Friend(token=token, backend=backend)
    bot.run()

if __name__ == "__main__":
    main()
