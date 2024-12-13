from virgil.friend.friend import Friend
import click

@click.command()
@click.option("--token", default=None, help="The token for the discord bot.")
@click.option("--backend", default="gemma", help="The backend to use for the chat.")
@click.option("--max-history-length", default=25, help="The maximum length of the chat history.")
@click.option("--prompt_filename", default="prompt.txt", help="The filename for the prompt.")
def main(token, backend, max_history_length, prompt_filename):
    bot = Friend(token=token, backend=backend, max_history_length=max_history_length, prompt_filename=prompt_filename)
    client = bot.client

    @bot.client.command(name="summon", help="Summon the bot to a channel.")
    async def summon(ctx):
        """Summon the bot to a channel."""
        print("Summoning the bot.")
        print(" -> Channel name:", ctx.channel.name)
        print(" -> Channel ID:", ctx.channel.id)
        bot.add_to_whitelist(ctx.channel.id)
        await ctx.send("Hello! I am here to help you.")

    bot.run()


if __name__ == "__main__":
    main()
