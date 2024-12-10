from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="virgil",
    version="0.1.0",
    author="Chris Paxton",
    author_email="chris.paxton.cs@gmail.com",
    description="A library for creating and using AI-powered chatbots and AI-powered storytelling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/virgil",
    packages=["virgil"],
    package_data={"virgil.quiz": ["quiz.html", "example_quiz.html"], "virgil.meme": ["prompt.txt"], "virgil.friend": ["prompt.txt"], "": ["config/*", "config/*/*"], "virgil.labyrinth": ["config/*"]},
    install_requires=[
        "transformers",
        "diffusers",
        "sentencepiece",
        "Pillow",
        "protobuf",
        "numpy",
        "click",
        "matplotlib",
        "discord.py",  # Install discord library for discord-using robots
        "python-dotenv",  # For pulling things like API keys and tokens from env
        "termcolor",  # For colored terminal output
        "hydra-core", # For configuration management
        "omegaconf", # For configuration management
    ],
    extras_require={
        "torch": [
            "torch>=2.5.1",
            "torchvision",
            "torchaudio",
            ],
        "dev": [
            "pytest",
            "flake8",
            "black",
            "isort",
            "mypy",
            "pre-commit",
            "codespell",
            "mdformat",
        ],
        "ai": [
            "torch>=2.5.1",
            "torchvision",
            "bitsandbytes",  # For quantization
            "accelerate",  # For model acceleration
            "flash-attn",
            # "xformers"
            # "autoawq",  # attention-aware quantization
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
