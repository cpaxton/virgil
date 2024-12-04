FROM python:3.8

WORKDIR /app

RUN apt-get install -y curl bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN curl -SLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -u
RUN pip install --upgrade pip
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN . "$HOME/.cargo/env"
RUN source "$HOME/.cargo/env"

COPY . .
RUN source "$HOME/.cargo/env"; pip install termcolor 'accelerate>=0.26.0'; PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 python -m pip install -e .;
RUN source "$HOME/.cargo/env"; pip install --pre -U xformers;
RUN export PATH="$HOME/miniconda3/bin:$PATH"; source "$HOME/.cargo/env"; cd /app; conda init bash;
RUN export PATH="$HOME/miniconda3/bin:$PATH"; source "$HOME/.cargo/env"; cd /app; conda create -n virgil python=3.8; eval "$(conda shell.bash hook)"; conda activate virgil;
CMD ["python", "-m", "virgil.friend.friend", "--backend", "qwen-1.5B"] 
