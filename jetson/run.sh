jetson-containers run --volume $HOME/src/virgil:/virgil --workdir /virgil   $(autotag l4t-text-generation) "pip3 install --user termcolor diffusers 
