import hydra
from omegaconf import DictConfig, OmegaConf

from virgil.labyrinth.maze import Maze
from virgil.backend import get_backend

class LabyrinthGenerator:
    """Load an LLM and use it to generate a labyrinth."""


@hydra.main(version_base=None, config_path="config", config_name="labyrinth")
def main(cfg: DictConfig):
    print("Application Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Create the maze to explore
    maze = Maze(cfg.maze.height, cfg.maze.width, seed=cfg.random_seed)
    backend = get_backend(cfg.backend)

if __name__ == "__main__":
    main()
