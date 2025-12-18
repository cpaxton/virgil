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

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image
import io


class MessageService(ABC):
    """
    Abstract service for messaging operations.
    This abstraction allows messaging functionality to be used
    by various agents (Discord bots, MCP servers, etc.) without tight coupling.
    """

    @abstractmethod
    async def send_message(
        self,
        content: str,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Send a text message to a channel or user.

        Args:
            content (str): The message content to send.
            channel_id (Optional[str]): The ID of the channel to send to.
            user_id (Optional[str]): The ID of the user to send a DM to.
                Either channel_id or user_id must be provided.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def post_in_channel(
        self,
        content: str,
        channel_id: str,
    ) -> bool:
        """
        Post a message in a specific channel.

        Args:
            content (str): The message content to post.
            channel_id (str): The ID of the channel to post to.

        Returns:
            bool: True if the message was posted successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def message_user(
        self,
        content: str,
        user_id: str,
    ) -> bool:
        """
        Send a direct message to a user.

        Args:
            content (str): The message content to send.
            user_id (str): The ID of the user to message.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def send_image(
        self,
        image: Image.Image,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> bool:
        """
        Send an image to a channel or user.

        Args:
            image (Image.Image): The PIL Image to send.
            channel_id (Optional[str]): The ID of the channel to send to.
            user_id (Optional[str]): The ID of the user to send a DM to.
                Either channel_id or user_id must be provided.
            caption (Optional[str]): Optional caption/text to send with the image.

        Returns:
            bool: True if the image was sent successfully, False otherwise.
        """
        pass


class DiscordMessageService(MessageService):
    """
    Discord-specific implementation of MessageService.
    """

    def __init__(self, discord_channel_getter, discord_user_getter):
        """
        Initialize the Discord message service.

        Args:
            discord_channel_getter: Function to get a Discord channel by ID.
            discord_user_getter: Function to get a Discord user by ID.
        """
        self._get_channel = discord_channel_getter
        self._get_user = discord_user_getter

    async def send_message(
        self,
        content: str,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Send a text message to a channel or user."""
        try:
            if channel_id:
                channel = self._get_channel(int(channel_id))
                if channel:
                    await channel.send(content)
                    return True
            elif user_id:
                user = self._get_user(int(user_id))
                if user:
                    await user.send(content)
                    return True
            return False
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    async def post_in_channel(self, content: str, channel_id: str) -> bool:
        """Post a message in a specific channel."""
        return await self.send_message(content, channel_id=channel_id)

    async def message_user(self, content: str, user_id: str) -> bool:
        """Send a direct message to a user."""
        return await self.send_message(content, user_id=user_id)

    async def send_image(
        self,
        image: Image.Image,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> bool:
        """Send an image to a channel or user."""
        try:
            import discord

            # Convert PIL Image to Discord file
            byte_arr = io.BytesIO()
            image.save(byte_arr, format="PNG")
            byte_arr.seek(0)
            file = discord.File(byte_arr, filename="image.png")

            if channel_id:
                channel = self._get_channel(int(channel_id))
                if channel:
                    if caption:
                        await channel.send(caption, file=file)
                    else:
                        await channel.send(file=file)
                    return True
            elif user_id:
                user = self._get_user(int(user_id))
                if user:
                    if caption:
                        await user.send(caption, file=file)
                    else:
                        await user.send(file=file)
                    return True
            return False
        except Exception as e:
            print(f"Error sending image: {e}")
            return False
