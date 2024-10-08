# Virgil

A simple set of tools and scripts for generating fun and compelling AI-generated content.

It's named after the AI from Mass Effect. Or maybe the Roman poet. Who can say.

## Installation

Just basic pip usage should do it:
```bash
python -m pip install -e .
```

You'll need to accept the [Gemma license](https://huggingface.co/google/gemma-2b). Virgil is designed to let you do text synethesis locally for weird and fun art projects, and Gemma is a great model for that since it can fit easily on a laptop GPU.

## Usage

The Infinite Quiz Machine requires about 12GB of GPU memory; I ran it on an NVIDIA RTX 4080 on my laptop and it worked fine. If that isn't available, you may need to do some tuning to get it to work on a smaller GPU.

### The Infinite Quiz Machine

After installing `virgil`:
```bash
./scripts/infinite_quiz_machine.sh "What kind of crustacean are you?"
```

Replace "What kind of crustacean are you?" with whatever prompt you want to use.

It will generate a quiz with 10 questions and answers, in a folder with the same name. It will also generate images using [diffusers](https://huggingface.co/docs/diffusers/en/index), with [SIGLip](https://huggingface.co/docs/transformers/en/model_doc/siglip) used to filter out lower-quality images.

Examples are available [on my website](https://cpaxton.github.io/quiz/). As it's an important topic I also have some [thoughts on AI art](https://itcanthink.substack.com/p/off-topic-what-role-for-ai-in-the) -- which is basically that I think it's a neat thing in its own right and not competitive with human art.

### Storyteller

After installing `virgil`:
```bash
python scripts/storyteller.py
```

This is a work in progress. Just chat with the world. See how it goes.

## License

If you somehow find this and want to use it just give me credit; everything here is by Chris Paxton
