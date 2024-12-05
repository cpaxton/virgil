import hydra
from hydra import utils
import os
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import numpy as np

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
        distances_to_goal = maze.compute_distances_from_goal()
        print("Distances:", distances)
        graph = maze.extract_graph()
        print("Graph:", graph)

        start_node = maze.get_start_point()
        print("Start point:", start_node)
        goal_node = maze.get_goal_point()
        print("Goal point:", goal_node)

        descriptions["world"] = {}
        descriptions["world"]["location"] = location
        descriptions["world"]["goal"] = goal
        descriptions["world"]["writing_style"] = writing_style
        descriptions["world"]["height"] = self.cfg.maze.height
        descriptions["world"]["width"] = self.cfg.maze.width
        descriptions["world"]["start"] = start_node
        descriptions["world"]["goal"] = goal_node

        for node, distance in distances.items():
            if node not in graph:
                next_nodes = []
            else:
                next_nodes = graph[node]
            print(node, "distance =", distance, "next nodes =", next_nodes)

            key = f"{node[0]}_{node[1]}"
            descriptions[key] = {}

            descriptions[key]["neighbors"] = [f"{n[0]}_{n[1]}" for n in next_nodes]
            descriptions[key]["distance"] = distance
            descriptions[key]["is_start"] = node[0] == start_node[0] and node[1] == start_node[1]
            descriptions[key]["is_goal"] = node[0] == goal_node[0] and node[1] == goal_node[1]
            descriptions[key]["is_dead_end"] = len(next_nodes) == 1
            descriptions[key]["is_junction"] = len(next_nodes) > 2

            # Inject some extra variation into the world
            if not descriptions[key]["is_start"] and not descriptions[key]["is_goal"]:
                r = np.random.rand()
                if r > 0.8:
                    descriptions[key]["has_npc"] = True
                else:
                    descriptions[key]["has_npc"] = False
                r = np.random.rand()
                if r > 0.8:
                    descriptions[key]["has_challenges"] = True
                else:
                    descriptions[key]["has_challenges"] = False
                r = np.random.rand()
                if r > 0.8:
                    descriptions[key]["is_unusual"] = True
                else:
                    descriptions[key]["is_unusual"] = False
            else:
                descriptions[key]["has_npc"] = False
                descriptions[key]["has_challenges"] = False
                descriptions[key]["is_unusual"] = False

            extra_info = ""
            if descriptions[key]["is_start"]:
                extra_info = " - This is the start room.\n"
            if descriptions[key]["is_goal"]:
                extra_info = f" - This is the goal room, containing {goal}\n"
            if descriptions[key]["is_dead_end"]:
                extra_info = " - This is a dead end. You must turn back.\n"
            if descriptions[key]["is_junction"]:
                extra_info = " - This is a junction. Multiple paths meet here.\n"
            if descriptions[key]["has_npc"]:
                extra_info = " - There is a non-player character here. They seem to be waiting for you. Describe them.\n"
            if descriptions[key]["has_challenges"]:
                extra_info = " - There is a challenge here. It will be hard to overcome. Describe it.\n"
            if descriptions[key]["is_unusual"]:
                extra_info = " - This place is unusual. It is not like the others. Describe why.\n"

            # Per room prompt filled out
            per_room_prompt = self.per_room_prompt.format(location=location, goal=goal, writing_style=writing_style, height=self.cfg.maze.height, width=self.cfg.maze.width, room=node, distance=distance, current_room=node, next_rooms=next_nodes, info=extra_info)
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
            if node == "world":
                continue
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
