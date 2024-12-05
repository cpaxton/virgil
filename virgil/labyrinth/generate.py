import hydra
from hydra import utils
import os
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from virgil.labyrinth.maze import Maze
from virgil.backend import get_backend
from virgil.chat import ChatWrapper

from virgil.image.diffuser import DiffuserImageGenerator


class LabyrinthGenerator:
    """Load an LLM and use it to generate a labyrinth."""

    def __init__(self, cfg: DictConfig, image_generator: Optional[DiffuserImageGenerator] = None) -> None:
        self.cfg = cfg

        # Set random seed
        if cfg.random_seed is not None:
            import random
            random.seed(cfg.random_seed)
            import numpy as np
            np.random.seed(cfg.random_seed)
            import torch
            torch.manual_seed(cfg.random_seed)

        # Create the backend
        self.backend = get_backend(self.cfg.backend)
        
        # Objects
        self.chat = ChatWrapper(backend=self.backend,
                                max_history_length=self.cfg.chat.max_history_length,
                                preserve=self.cfg.chat.messages_to_preserve)
        self.image_promt_generator = ChatWrapper(backend=self.backend,
                                                 max_history_length=self.cfg.chat.max_history_length,
                                                 preserve=self.cfg.chat.messages_to_preserve)

        # Load the prompt
        self.initial_prompt_template = self.load_prompt(cfg.prompt.initial_prompt)
        self.per_room_prompt = self.load_prompt(cfg.prompt.per_room_prompt)
        self.image_prompt = self.load_prompt(cfg.prompt.image_prompt)

        if image_generator is None:
            self.image_generator = DiffuserImageGenerator(height=512, width=512, num_inference_steps=4, guidance_scale=0.0, model="turbo", xformers=False)
        else:
            self.image_generator = image_generator

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

        # Hold all the prompts
        descriptions = {}
        
        # compute distances from start for everything
        distances = maze.compute_distances_from_start()
        print("Distances:", distances)
        graph = maze.extract_graph()
        print("Graph:", graph)

        print("Start point:", maze.get_start_point())
        print("Goal point:", maze.get_goal_point())

        for node, distance in distances.items():
            if node not in graph:
                next_nodes = []
            else:
                next_nodes = graph[node]
            print(node, "distance =", distance, "next nodes =", next_nodes)

            key = f"{node[0]}_{node[1]}"
            descriptions[key] = {}

            descriptions[key]["location"] = location
            descriptions[key]["goal"] = goal
            descriptions[key]["writing_style"] = writing_style
            descriptions[key]["height"] = self.cfg.maze.height
            descriptions[key]["width"] = self.cfg.maze.width
            descriptions[key]["neighbors"] = [f"{n[0]}_{n[1]}" for n in next_nodes]
            descriptions[key]["distance"] = distance
            descriptions[key]["is_start"] = node == maze.get_start_point()
            descriptions[key]["is_goal"] = node == maze.get_goal_point()

            # Per room prompt filled out
            per_room_prompt = self.per_room_prompt.format(location=location, goal=goal, writing_style=writing_style, height=self.cfg.maze.height, width=self.cfg.maze.width, room=node, distance=distance, current_room=node, next_rooms=next_nodes)
            description = self.chat.prompt(per_room_prompt, verbose=True)
            descriptions[key]["text"] = description

            # Generate the image prompt
            image_prompt = self.image_prompt.format(location=location, description=description)
            image_description = self.image_promt_generator.prompt(image_prompt, verbose=True)
            descriptions[key]["image"] = image_description
            
            descriptions[key]["image_filename"] = f"{node[0]}_{node[1]}.png"

        # Create folder based on location nmae
        folder_name = location.replace(" ", "_").lower()
        os.makedirs(folder_name, exist_ok=True)

        # Create yaml dump of the descriptions
        with open(os.path.join(folder_name, "descriptions.yaml"), "w") as f:
            OmegaConf.save(descriptions, f)

        # Generate images for each room
        for node, description in descriptions.items():
            image_filename = os.path.join(folder_name, description["image_filename"])
            self.generate_image(description["image"], image_filename)

    def generate_image(self, image_description: str, image_filename: str) -> None:

        # Generate the image
        image = self.image_generator.generate(self.cfg.world.image_style + " " + image_description)
        image.save(image_filename)

        print(f"Saved image to {image_filename}")


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
