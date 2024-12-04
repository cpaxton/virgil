
# Discord Setup Gudie

So you want to create a Discord chatbot in python? You've come to the right place, perhaps to use the Virgil chatbot ("friend"). This guide will walk you through the process of creating a Discord bot and getting it set up to run on your server.

## Create a Discord Bot

You will need to create your Discord bot and get a token. You can do this by going to the [Discord Developer Portal](https://discord.com/developers/applications) and creating a new application. Then, create a bot and get the token.

You can find the token in the "Installation" tab of the discord developer UI for your app. You will be able to get a link something like this:

```
https://discord.com/oauth2/authorize?client_id=999999999999999999&scope=bot
```

But you'll need to replace the `client_id` with your own:

```
https://discord.com/oauth2/authorize?client_id=$TOKEN&scope=bot
```

where `$TOKEN` is the client ID of your bot. This can be found either in the provided URL on the "Installation" page, or explicitly on the "Oath2" page.

## Installation Page

![Installation Page](images/install_page.png)

Here, you need to get the Discord-provided installation link. It will look something like the link below, but you'll need to replace the `client_id` with your own:

```
https://discord.com/oauth2/authorize?client_id=$TOKEN
```

where `$TOKEN` is the client ID of your bot. This can be found either in the provided URL on the "Installation" page, or explicitly on the "Oath2" page.

## Set Permissions in OAuth2 and Create an Install Link

Then you need to set permissions for the bot properly and create an install link. For more detailed instructions, see the [Discord setup guide](docs/discord.md).

First make sure redirects is populated with the URL from Installation:

![Redirects](images/oauth2_redirects.png)

Then make sure the bot has the correct permissions. You'll need `guild`, `messages.read`, and `bot` permissions at a minimum. Then, choose a redict URL (the one you just entered above), and add the correct Bot permissions as well.

The bot permissions need to include sending files and messages.

![Bot Permissions](images/bot_permissions.png)

Then you can create an install link. This will be the link you use to add the bot to your server. Integration type should be "guild install" -- guild is the internal term for a server in the Discord API.

Finally, you'll get a URL you can copy into Discord and use to install your bot in a server.

![Install Example](images/add_bot_example.png)

It should look something like this (depending on your bot's name).
