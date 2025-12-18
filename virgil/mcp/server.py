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

"""
MCP (Model Context Protocol) server for Virgil.
Exposes Virgil's capabilities (image generation, messaging, etc.) as MCP tools
for use by general-purpose AI agents.
"""

import asyncio
import io
from typing import Optional

try:
    # Try different possible import paths for MCP
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent, ImageContent
    except ImportError:
        # Alternative import path
        from mcp import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent, ImageContent
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_AVAILABLE = False
    _mcp_import_error = e

    # Create dummy classes for type checking when MCP is not available
    class Server:
        pass

    def stdio_server():
        raise ImportError(
            f"MCP library is not installed. Install it with: uv sync\n"
            f"This will install all dependencies including MCP from pyproject.toml\n"
            f"Original error: {_mcp_import_error}"
        )


from virgil.services.image_service import ImageService
from virgil.services.message_service import MessageService


class VirgilMCPServer:
    """
    MCP server that exposes Virgil's capabilities as tools.
    """

    def __init__(
        self,
        image_service: Optional[ImageService] = None,
        message_service: Optional[MessageService] = None,
    ):
        """
        Initialize the MCP server.

        Args:
            image_service (Optional[ImageService]): Service for image generation.
            message_service (Optional[MessageService]): Service for messaging operations.
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP library is not installed. Install it with: uv sync\n"
                "This will install all dependencies including MCP from pyproject.toml"
            )

        self.image_service = image_service
        self.message_service = message_service
        self.server = Server("virgil")

        self._setup_tools()

    def _setup_tools(self):
        """Register MCP tools with the server."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            tools = []

            if self.image_service:
                tools.append(
                    Tool(
                        name="generate_image",
                        description="Generate an image from a text prompt using AI image generation.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The text description of the image to generate.",
                                }
                            },
                            "required": ["prompt"],
                        },
                    )
                )

            if self.message_service:
                tools.append(
                    Tool(
                        name="send_message",
                        description="Send a text message to a channel or user.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The message content to send.",
                                },
                                "channel_id": {
                                    "type": "string",
                                    "description": "The ID of the channel to send to (optional if user_id is provided).",
                                },
                                "user_id": {
                                    "type": "string",
                                    "description": "The ID of the user to send a DM to (optional if channel_id is provided).",
                                },
                            },
                            "required": ["content"],
                        },
                    )
                )

                tools.append(
                    Tool(
                        name="post_in_channel",
                        description="Post a message in a specific channel.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The message content to post.",
                                },
                                "channel_id": {
                                    "type": "string",
                                    "description": "The ID of the channel to post to.",
                                },
                            },
                            "required": ["content", "channel_id"],
                        },
                    )
                )

                tools.append(
                    Tool(
                        name="message_user",
                        description="Send a direct message to a user.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The message content to send.",
                                },
                                "user_id": {
                                    "type": "string",
                                    "description": "The ID of the user to message.",
                                },
                            },
                            "required": ["content", "user_id"],
                        },
                    )
                )

                tools.append(
                    Tool(
                        name="send_image",
                        description="Send an image to a channel or user.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_prompt": {
                                    "type": "string",
                                    "description": "The text prompt to generate an image from, or 'use_last_generated' to use the last generated image.",
                                },
                                "channel_id": {
                                    "type": "string",
                                    "description": "The ID of the channel to send to (optional if user_id is provided).",
                                },
                                "user_id": {
                                    "type": "string",
                                    "description": "The ID of the user to send a DM to (optional if channel_id is provided).",
                                },
                                "caption": {
                                    "type": "string",
                                    "description": "Optional caption/text to send with the image.",
                                },
                            },
                            "required": ["image_prompt"],
                        },
                    )
                )

            return tools

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: dict
        ) -> list[TextContent | ImageContent]:
            """Handle tool calls."""
            if name == "generate_image" and self.image_service:
                prompt = arguments.get("prompt")
                if not prompt:
                    return [
                        TextContent(
                            type="text", text="Error: 'prompt' parameter is required."
                        )
                    ]

                try:
                    image = self.image_service.generate_image(prompt)
                    # Store the last generated image for potential reuse
                    self._last_generated_image = image

                    # Convert image to base64 for MCP
                    import base64

                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    return [
                        TextContent(
                            type="text",
                            text=f"Successfully generated image from prompt: {prompt}",
                        ),
                        ImageContent(
                            type="image",
                            data=img_base64,
                            mimeType="image/png",
                        ),
                    ]
                except Exception as e:
                    return [
                        TextContent(
                            type="text", text=f"Error generating image: {str(e)}"
                        )
                    ]

            elif name == "send_message" and self.message_service:
                content = arguments.get("content")
                channel_id = arguments.get("channel_id")
                user_id = arguments.get("user_id")

                if not content:
                    return [
                        TextContent(
                            type="text", text="Error: 'content' parameter is required."
                        )
                    ]
                if not channel_id and not user_id:
                    return [
                        TextContent(
                            type="text",
                            text="Error: Either 'channel_id' or 'user_id' must be provided.",
                        )
                    ]

                try:
                    success = await self.message_service.send_message(
                        content, channel_id=channel_id, user_id=user_id
                    )
                    if success:
                        return [
                            TextContent(
                                type="text",
                                text=f"Successfully sent message to {'channel' if channel_id else 'user'}.",
                            )
                        ]
                    else:
                        return [
                            TextContent(type="text", text="Failed to send message.")
                        ]
                except Exception as e:
                    return [
                        TextContent(
                            type="text", text=f"Error sending message: {str(e)}"
                        )
                    ]

            elif name == "post_in_channel" and self.message_service:
                content = arguments.get("content")
                channel_id = arguments.get("channel_id")

                if not content or not channel_id:
                    return [
                        TextContent(
                            type="text",
                            text="Error: Both 'content' and 'channel_id' parameters are required.",
                        )
                    ]

                try:
                    success = await self.message_service.post_in_channel(
                        content, channel_id
                    )
                    if success:
                        return [
                            TextContent(
                                type="text",
                                text=f"Successfully posted message in channel {channel_id}.",
                            )
                        ]
                    else:
                        return [
                            TextContent(type="text", text="Failed to post message.")
                        ]
                except Exception as e:
                    return [
                        TextContent(
                            type="text", text=f"Error posting message: {str(e)}"
                        )
                    ]

            elif name == "message_user" and self.message_service:
                content = arguments.get("content")
                user_id = arguments.get("user_id")

                if not content or not user_id:
                    return [
                        TextContent(
                            type="text",
                            text="Error: Both 'content' and 'user_id' parameters are required.",
                        )
                    ]

                try:
                    success = await self.message_service.message_user(content, user_id)
                    if success:
                        return [
                            TextContent(
                                type="text",
                                text=f"Successfully sent message to user {user_id}.",
                            )
                        ]
                    else:
                        return [
                            TextContent(type="text", text="Failed to send message.")
                        ]
                except Exception as e:
                    return [
                        TextContent(
                            type="text", text=f"Error sending message: {str(e)}"
                        )
                    ]

            elif name == "send_image" and self.message_service:
                image_prompt = arguments.get("image_prompt")
                channel_id = arguments.get("channel_id")
                user_id = arguments.get("user_id")
                caption = arguments.get("caption")

                if not image_prompt:
                    return [
                        TextContent(
                            type="text",
                            text="Error: 'image_prompt' parameter is required.",
                        )
                    ]
                if not channel_id and not user_id:
                    return [
                        TextContent(
                            type="text",
                            text="Error: Either 'channel_id' or 'user_id' must be provided.",
                        )
                    ]

                try:
                    # Generate or retrieve image
                    if image_prompt == "use_last_generated" and hasattr(
                        self, "_last_generated_image"
                    ):
                        image = self._last_generated_image
                    elif self.image_service:
                        image = self.image_service.generate_image(image_prompt)
                        self._last_generated_image = image
                    else:
                        return [
                            TextContent(
                                type="text",
                                text="Error: Image service is not available.",
                            )
                        ]

                    success = await self.message_service.send_image(
                        image, channel_id=channel_id, user_id=user_id, caption=caption
                    )
                    if success:
                        return [
                            TextContent(
                                type="text",
                                text=f"Successfully sent image to {'channel' if channel_id else 'user'}.",
                            )
                        ]
                    else:
                        return [TextContent(type="text", text="Failed to send image.")]
                except Exception as e:
                    return [
                        TextContent(type="text", text=f"Error sending image: {str(e)}")
                    ]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # Initialize last generated image storage
        self._last_generated_image = None

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main():
    """Main entry point for running the MCP server standalone."""
    # This would typically be configured by the user
    # For now, we'll create a minimal example
    server = VirgilMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
