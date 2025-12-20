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
Example usage of the Virgil MCP server.

This example shows how to set up and run the MCP server with Virgil's
image generation and messaging capabilities.
"""

import asyncio
from virgil.mcp.server import VirgilMCPServer
from virgil.services.image_service import VirgilImageService
from virgil.image import DiffuserImageGenerator


async def main():
    """Example of running the MCP server."""
    # Initialize image service
    image_generator = DiffuserImageGenerator(
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=0.0,
        model="turbo",
        xformers=False,
    )
    image_service = VirgilImageService(image_generator)

    # Note: For a full example, you would also initialize a message_service
    # For Discord, use DiscordMessageService from virgil.services.message_service
    # For other platforms, implement your own MessageService subclass

    # Create and run the MCP server
    server = VirgilMCPServer(
        image_service=image_service,
        message_service=None,  # Set to your message service if available
    )

    print("Starting Virgil MCP server...")
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
