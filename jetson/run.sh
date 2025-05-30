#!/usr/bin/env bash
# Starts program
# Installs some dependencies
# Passes in the discord token
jetson-containers run --volume $HOME/src/virgil:/virgil --workdir /virgil  -e DISCORD_TOKEN=$DISCORD_TOKEN $(autotag l4t-text-generation)
# pip3 install --user termcolor diffusers discord.py && export PYTHONPATH=/virgil:$PYTHONPATH && python3 -m virgil.friend.friend --backend qwen

# jetson-containers run --volume $HOME/src/virgil:/virgil --workdir /virgil  -e DISCORD_TOKEN=$DISCORD_TOKEN $(autotag l4t-text-generation)
# pip3 install --upgrade transformers && pip3 install --user termcolor diffusers discord.py && export PYTHONPATH=/virgil:$PYTHONPATH && python3 -m virgil.friend.friend"
