import hydra
from hydra import utils
import os
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from virgil.labyrinth.maze import Maze
from virgil.backend import get_backend
from virgil.chat import ChatWrapper

class LabyrinthGenerator:
    """Load an LLM and use it to generate a labyrinth."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Create the backend
        self.backend = get_backend(self.cfg.backend)
        
        # Objects
        self.chat = ChatWrapper(backend=self.backend,
                                max_history_length=self.cfg.chat.max_history_length,
                                preserve=self.cfg.chat.messages_to_preserve)

        # Load the prompt
        self.initial_prompt_template = self.load_prompt(cfg.prompt.initial_prompt)

    def load_prompt(self, prompt: str) -> str:
        """Load a prompt from a file or string."""
        prompt_filename = os.path.join(self.cfg.prompt.prompt_dir, prompt)
        if os.path.isfile(prompt_filename):
            with open(prompt_filename, "r") as f:
                text = f.read()
                parsed_text = ""
                for line in text.splitlines():
                    if line is None:
                        continue
                    elif len(line) == 0:
                        parsed_text += "\n"
                    elif len(line) > 0 and line.strip() [0] != "#":
                        parsed_text += line + "\n"
                return parsed_text
        else:
            return prompt

    def create_maze(self) -> Maze:

        # Create the maze to explore
        maze = Maze(self.cfg.maze.height, self.cfg.maze.width, seed=self.cfg.random_seed)
        graph = maze.extract_graph()

        # TODO: remove debug info
        # print(graph)

        if self.cfg.maze.visualize:
            maze.draw_maze_with_graph()

        return maze

    def generate(self, location: Optional[str] = None, goal: Optional[str] = None, writing_style: Optional[str] = None) -> None:
        self.chat.clear()

        if goal is None:
            goal = self.cfg.world.goal
        if location is None:
            location = self.cfg.world.location
        if writing_style is None:
            writing_style = self.cfg.world.writing_style

        initial_prompt = self.initial_prompt_template.format(location=location, goal=goal, writing_style=writing_style, height=self.cfg.maze.height, width=self.cfg.maze.width)

        res = self.chat.prompt(initial_prompt, verbose=True)
        assert "acknowledge" in res.lower(), "Failed to acknowledge the initial prompt."
    
        # Generate a random maze
        maze = self.create_maze()


@hydra.main(version_base=None, config_path="config", config_name="labyrinth")
def main(cfg: DictConfig):
    print("Application Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Create a LabyrinthGenerator
    labyrinth_generator = LabyrinthGenerator(cfg)

    # Generate the default labyrinth
    labyrinth_generator.generate()

if __name__ == "__main__":
    main()
