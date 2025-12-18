# Virgil MCP Server

This module provides MCP (Model Context Protocol) support for Virgil, allowing general-purpose AI agents to use Virgil's capabilities (image generation, messaging, etc.) through the MCP protocol.

## Installation

MCP is now included as a default dependency. Install all dependencies using `uv sync`:

```bash
uv sync
```

Note: This project uses `uv` for package management. Make sure your virtual environment is activated:
```bash
source .venv/bin/activate
```

## Usage

### Running Friend as MCP Server

The easiest way to use Virgil's MCP server is through the Friend CLI with the `--mcp-only` flag:

```bash
python -m virgil.friend.friend --mcp-only --image-generator diffuser
```

This runs Friend as a standalone MCP server, exposing image generation capabilities via MCP protocol. The server communicates via stdin/stdout and can be invoked by MCP clients.

### Running Friend with Discord + MCP

You can also run Friend as a Discord bot with MCP support enabled (experimental):

```bash
python -m virgil.friend.friend --enable-mcp --token YOUR_DISCORD_TOKEN
```

### Programmatic Usage

```python
import asyncio
from virgil.mcp.server import VirgilMCPServer
from virgil.services.image_service import VirgilImageService
from virgil.image import DiffuserImageGenerator

async def main():
    # Initialize image service
    image_generator = DiffuserImageGenerator(
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=0.0,
        model="turbo",
    )
    image_service = VirgilImageService(image_generator)

    # Create MCP server
    server = VirgilMCPServer(image_service=image_service)
    
    # Run the server
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### With Discord Messaging

```python
from virgil.services.message_service import DiscordMessageService

# In your Discord bot's on_ready():
def get_channel(channel_id: int):
    return client.get_channel(channel_id)

def get_user(user_id: int):
    return client.get_user(user_id)

message_service = DiscordMessageService(
    discord_channel_getter=get_channel,
    discord_user_getter=get_user,
)

server = VirgilMCPServer(
    image_service=image_service,
    message_service=message_service,
)
```

## Available Tools

The MCP server exposes the following tools:

1. **generate_image**: Generate an image from a text prompt
2. **send_message**: Send a text message to a channel or user
3. **post_in_channel**: Post a message in a specific channel
4. **message_user**: Send a direct message to a user
5. **send_image**: Generate and send an image to a channel or user

## Architecture

The MCP server uses a service-oriented architecture:

- **ImageService**: Abstract interface for image generation
- **MessageService**: Abstract interface for messaging operations
- **VirgilImageService**: Implementation using Virgil's image generators
- **DiscordMessageService**: Discord-specific messaging implementation

This design allows the same functionality to be used by:
- Discord bots (via `Friend` class)
- MCP servers (via `VirgilMCPServer`)
- Other AI agents (by implementing custom services)
