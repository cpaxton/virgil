# Virgil AI

A simple set of tools and scripts for generating fun and compelling AI-generated content.

It's named after the AI from Mass Effect. Or maybe the Roman poet.

[Check the installation guide](docs/install.md) for more information on how to install and use Virgil.

## Usage

- [Chat](#chat): A simple chatbot test with no prompt. The raw LLM experience, useful for development.
- [The Infinite Quiz Machine](#the-infinite-quiz-machine): Generates a quiz based on a prompt.
- [Friend](#friend): A chatbot which works with Discord, can use different LLM backends, generate images, and has a (somewhat) persistent memory.
- [Storyteller](#storyteller): A work-in-progress of a single scene at the "beginning" of a cosmic horror story.

The Infinite Quiz Machine requires about 12GB of GPU memory; I ran it on an NVIDIA RTX 4080 on my laptop and it worked fine. If that isn't available, you may need to do some tuning to get it to work on a smaller GPU.

Bots like Friend can run on a smaller GPU, but you may need to adjust the batch size to get it to work; they'll die if they run out of memory, and so you may want to restart them. Nothing clever is being done here to keep them to a certain memory limit.

### Chat

Simple chatbot test, useful for playing around with different LLMs ("backends") in a chat setting before deploying them in more complex projects.

After installing `virgil`:

```bash
python -m virgil.chat
```

You can also provide a backend, e.g.

```bash
python -m virgil.chat --backend qwen-1.5B
```

This application is useful for development. You can provide a prompt as a text file in order to see how it works, e.g.:

```bash
python virgil/chat.py --backend qwen-1.5B --prompt virgil/labyrinth/config/prompt_castle.txt
```

will load the initial prompt from [Labyrinth](#labyrinth) and use the Qwen 1.5B model to generate text. You can then continue to interrogate it in dialogue.

### The Infinite Quiz Machine

After installing `virgil`:

```bash
./scripts/infinite_quiz_machine.sh "What kind of crustacean are you?"
```

Replace "What kind of crustacean are you?" with whatever prompt you want to use.

It will generate a quiz with 10 questions and answers, in a folder with the same name. It will also generate images using [diffusers](https://huggingface.co/docs/diffusers/en/index), with [SIGLip](https://huggingface.co/docs/transformers/en/model_doc/siglip) used to filter out lower-quality images.

Examples are available [on my website](https://cpaxton.github.io/quiz/). Not all of them are winners, but plenty of them are fun. As it's an important and related topic, I also have some [thoughts on AI art](https://itcanthink.substack.com/p/off-topic-what-role-for-ai-in-the) -- which is basically that I think it's a neat thing in its own right and not competitive with human art.

### Friend

Friend is a simple chatbot which can use various backends to generate text on Discord. After installing `virgil`:

```bash
python -m virgil.friend.friend

# You can use various backends
# Gemma 2B is the default, small and fast, but not the fastest
python -m virgil.friend.friend --backend gemma

# Qwen 1.5B is a small model and should work on a laptop GPU
python -m virgil.friend.friend --backend qwen-1.5B
```

#### Discord Setup for Friend

You will need to create your Discord bot and get a token. You can do this by going to the [Discord Developer Portal](https://discord.com/developers/applications) and creating a new application. Then, create a bot and get the token.

You can find the token in the "Installation" tab of the discord developer UI for your app. You will be able to get a link something like this:

```
https://discord.com/oauth2/authorize?client_id=999999999999999999&scope=bot
```

But you'll need to replace the `client_id` with your own:

```
https://discord.com/oauth2/authorize?client_id=$TOKEN&scope=bot
```

where `$TOKEN` is the client ID of your bot. This can be found either in the provided URL on the "Installation" page, or explicitly on the "Oath2" page. Then you need to set permissions for the bot properly and create an install link. For more detailed instructions, see the [Discord setup guide](docs/discord.md).

### Labyrinth

Generate a maze you can explore with associated text and images. This is a work-in-progress.

Inputs:
  - the "quest" you're on -- this is a prompt which will be used to generate the maze. It also determines the end goal of the maze.
  - the  location you're in -- a castle, a forest, etc. Something about what kind of world you want to generate.

### Storyteller

After installing `virgil`:

```bash
python -m virgil.story.cosmic_horror
```

This is a work-in-progress of a single scene at the "beginning" of a cosmic horror story. You've just arrived at a bus stop in the town of Greenwood, Ohio, looking for your sister. See what you can find out.

## License

If you somehow find this and want to use it, please give me credit; Virgil is a project by [Chris Paxton](https://cpaxton.github.io/).

The code is covered by an [Apache 2.0 license](LICENSE), but the models and data are covered by their own licenses.
