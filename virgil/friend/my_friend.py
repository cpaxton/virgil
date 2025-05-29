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

from virgil.friend.friend import Friend
import click


@click.command()
@click.option("--token", default=None, help="The token for the discord bot.")
@click.option("--backend", default="gemma", help="The backend to use for the chat.")
@click.option(
    "--max-history-length", default=25, help="The maximum length of the chat history."
)
def main(token, backend, max_history_length):
    prompt_filename = "my_prompt.txt"
    bot = Friend(
        token=token,
        backend=backend,
        max_history_length=max_history_length,
        prompt_filename=prompt_filename,
    )

    @bot.client.command(name="summon", help="Summon the bot to a channel.")
    async def summon(ctx):
        """Summon the bot to a channel."""
        print("Summoning the bot.")
        print(" -> Channel name:", ctx.channel.name)
        print(" -> Channel ID:", ctx.channel.id)
        bot.allowed_channels.visit(ctx.channel)
        await ctx.send("Hello! I am here to help you.")

    bot.run()


if __name__ == "__main__":
    main()
