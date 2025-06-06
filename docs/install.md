# Virgil AI

## Installation

If installing locally, it's highly recommended to create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for this project:

```bash
conda create -n virgil python=3.10
conda activate virgil

# On an NVIDIA jetson it might be easier to install on an earlier version of python
conda create -n virgil python=3.8
```

Then install pytorch and torchvision. You can find the right command for your system [here](https://pytorch.org/get-started/locally/).

For example:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

After that, just a basic pip usage should do it:

```bash
python -m pip install -e .[ai]
```

The `[ai]` flag will install some acceleration tools for AI models, like [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), which may cause issues on some systems. If you have trouble, you can remove it.

To use Gemma, you'll need to accept the [Gemma license](https://huggingface.co/google/gemma-2b). Virgil is designed to let you do text synethesis locally for weird and fun art projects, and Gemma is a great model for that since it can fit easily on a laptop GPU.

### Optional: bitsandbytes

You may want to use [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), which is a tool for accelerating AI models. You can install it with:

```bash
pip install -U bitsandbytes
```

### Optional: `flash_attn2`

Install flash-attn2 for faster attention:

```bash
pip install flash-attn --no-build-isolation
```

*Note as of 2025-01-18:* There's a compatibility issue between the `flash_attn` package and the latest version of `transformers==4.48`, at least in the Google Gemma LLM. Instead of using the latest version of `transformers`, stick with an older version:
```bash
python -m pip install transformers==4.46.3
```

### Optional: xformers

You may want to use [xformers](https://github.com/facebookresearch/xformers) for better memory efficiency. You can install it with:

```bash
pip install --pre -U xformers
```

Or if you followed the conda instructions above:

```bash
conda install xformers -c xformers
```
