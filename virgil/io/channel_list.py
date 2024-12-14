from itertools import chain
from typing import List, Dict
import discord


class ChannelList:
    """Tracks the channels that the bot is allowed to post in."""

    def __init__(self):
        self.home_channels: List[discord.TextChannel] = []
        self.visiting_channels: Dict[discord.TextChannel, float] = {} # Maps channel to expiration time

    def add_home(self, channel: discord.TextChannel):
        """Add a channel to the home list."""
        self.home_channels.append(channel)

    def remove_home(self, channel: discord.TextChannel):
        """Remove a channel from the home list."""
        self.home_channels = [c for c in self.home_channels if c != channel]

    def visit(self, channel: discord.TextChannel, timeout_s: float = 60):
        """Add a channel to the visiting list."""
        self.visiting_channels[channel] = timeit.default_timer() + timeout_s

    def is_valid(self, channel: discord.TextChannel):
        """Check if a channel is valid to post in."""
        if channel in self.home_channels:
            return True
        elif expiration := self.visiting_channels.get(channel):
            if expiration > timeit.default_timer():
                return True
            print(f"Visit to channel has expired. {channel}")
            del self.visiting_channels[channel]
        return False

    def __contains__(self, channel: discord.TextChannel):
        return self.is_valid(channel)

    def __iter__(self):
        return chain(self.home_channels, (vc.channel for vc in self.visiting_channels.values()))

    def __len__(self):
        return len(self.home_channels) + len(self.visiting_channels)


    def __str__(self) -> str:
        return f"Home: {self.home_channels}\nVisiting: {self.visiting_channels.keys()}"
