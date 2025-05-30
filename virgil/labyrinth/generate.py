# Copyright 2024 Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import os
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import numpy as np
import shutil

from virgil.labyrinth.maze import Maze
from virgil.backend import get_backend
from virgil.chat import ChatWrapper
from virgil.utils import load_prompt

from virgil.image.diffuser import DiffuserImageGenerator


# Get the path to config/index.html
def get_index_html_path() -> str:
    return os.path.join(os.path.dirname(__file__), "config", "index.html")


def get_labyrinth_js_path() -> str:
    return os.path.join(os.path.dirname(__file__), "config", "labyrinth.js")


class LabyrinthGenerator:
    """Load an LLM and use it to generate a labyrinth."""

    def __init__(
        self, cfg: DictConfig, image_generator: Optional[DiffuserImageGenerator] = None
    ) -> None:
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
        self.chat = ChatWrapper(
            backend=self.backend,
            max_history_length=self.cfg.chat.max_history_length,
            preserve=self.cfg.chat.messages_to_preserve,
        )
        self.image_promt_generator = ChatWrapper(
            backend=self.backend,
            max_history_length=self.cfg.chat.max_history_length,
            preserve=self.cfg.chat.messages_to_preserve,
        )

        # Load the prompt
        self.initial_prompt_template = self.load_prompt(cfg.prompt.initial_prompt)
        self.per_room_prompt = self.load_prompt(cfg.prompt.per_room_prompt)
        self.image_prompt = self.load_prompt(cfg.prompt.image_prompt)

        if image_generator is None:
            self.image_generator = DiffuserImageGenerator(
                height=512,
                width=512,
                num_inference_steps=4,
                guidance_scale=0.0,
                model="turbo",
                xformers=False,
            )
        else:
            self.image_generator = image_generator

    def load_prompt(self, prompt: str) -> str:
        """Load a prompt from a file or string."""
        prompt_filename = os.path.join(self.cfg.prompt.prompt_dir, prompt)
        return load_prompt(prompt_filename)

    def create_maze(self) -> Maze:
        # Create the maze to explore
        maze = Maze(
            self.cfg.maze.height, self.cfg.maze.width, seed=self.cfg.random_seed
        )

        # TODO: remove debug info
        # graph = maze.extract_graph()
        # print(graph)

        if self.cfg.maze.visualize:
            maze.draw_maze_with_graph()

        return maze

    def generate(
        self,
        location: Optional[str] = None,
        goal: Optional[str] = None,
        writing_style: Optional[str] = None,
        image_style: Optional[str] = None,
    ) -> None:
        """
        Generate a labyrinth based on the given parameters.

        Args:
            location: The location of the labyrinth.
            goal: The goal of the labyrinth.
            writing_style: The writing style to use.
            image_style: The image style to use.
        """
        self.chat.clear()

        if goal is None:
            goal = self.cfg.world.goal
        if location is None:
            location = self.cfg.world.location
        if writing_style is None:
            writing_style = self.cfg.world.writing_style
        if image_style is None:
            image_style = self.cfg.world.image_style

        initial_prompt = self.initial_prompt_template.format(
            location=location,
            goal=goal,
            writing_style=writing_style,
            height=self.cfg.maze.height,
            width=self.cfg.maze.width,
        )

        res = self.chat.prompt(initial_prompt, verbose=True)
        assert "acknowledge" in res.lower(), "Failed to acknowledge the initial prompt."

        # Generate a random maze
        maze = self.create_maze()

        # Hold all the prompts
        descriptions = {}

        # compute distances from start for everything
        distances = maze.compute_distances_from_start()
        # TODO: remove debug info
        # distances_to_goal = maze.compute_distances_from_goal()
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
            descriptions[key]["is_start"] = (
                node[0] == start_node[0] and node[1] == start_node[1]
            )
            descriptions[key]["is_goal"] = (
                node[0] == goal_node[0] and node[1] == goal_node[1]
            )
            descriptions[key]["is_dead_end"] = len(next_nodes) == 1
            descriptions[key]["is_junction"] = len(next_nodes) > 2

            # Start and goal are not dead ends
            if node[0] == start_node[0] and node[1] == start_node[1]:
                descriptions[key]["is_start"] = True
                descriptions[key]["is_dead_end"] = False
            if node[0] == goal_node[0] and node[1] == goal_node[1]:
                descriptions[key]["is_goal"] = True
                descriptions[key]["is_dead_end"] = False

            # Inject some extra variation into the world
            descriptions[key]["has_npc"] = False
            descriptions[key]["has_challenges"] = False
            descriptions[key]["is_unusual"] = False
            if not descriptions[key]["is_start"] and not descriptions[key]["is_goal"]:
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                if r1 > 0.5:
                    descriptions[key]["has_npc"] = True
                elif r2 > 0.5:
                    descriptions[key]["has_challenges"] = True
                elif r3 > 0.5:
                    descriptions[key]["is_unusual"] = True

            extra_info = ""
            if descriptions[key]["is_start"]:
                extra_info = " - This is the start room.\n"
            if descriptions[key]["is_goal"]:
                extra_info = (
                    f" - This is the goal room, containing {goal}. Describe it.\n"
                )
            if descriptions[key]["is_dead_end"]:
                extra_info = " - This is a dead end. Describe it.\n"
            if descriptions[key]["is_junction"]:
                extra_info = " - Multiple paths meet here. Describe them.\n"
            if descriptions[key]["has_npc"]:
                extra_info = (
                    " - There is someone here, waiting for you. Describe them.\n"
                )
            if descriptions[key]["has_challenges"]:
                extra_info = " - There is a challenge here that will be hard to overcome. Describe it.\n"
            if descriptions[key]["is_unusual"]:
                extra_info = " - This place is unusual. It is not like the others. Describe why.\n"

            # Per room prompt filled out
            per_room_prompt = self.per_room_prompt.format(
                location=location,
                goal=goal,
                writing_style=writing_style,
                height=self.cfg.maze.height,
                width=self.cfg.maze.width,
                room=node,
                distance=distance,
                current_room=node,
                next_rooms=next_nodes,
                info=extra_info,
            )
            description = self.chat.prompt(per_room_prompt, verbose=True)
            descriptions[key]["text"] = description

            # Set the title
            room_title = self.chat.prompt(
                'Name this room. Name should be concise, < 5 words, and descriptive, e.g. "The Great Hall," "Secluded Clearing." It should not contain {node}. Name:',
                verbose=False,
            )
            print("Room title:", room_title)
            descriptions[key]["title"] = room_title

            if descriptions[key]["has_npc"]:
                npc_description = self.chat.prompt(
                    "Describe appearance of the the NPC in this room. The NPC should be a character that the player can interact with. Do not use the term NPC. Describe them in a few sentences, starting with their appearance."
                )
                print("NPC description:", npc_description)
                descriptions[key]["npc"] = npc_description
                descriptions[key]["text"] += f"\n\n{npc_description}"
            elif descriptions[key]["has_challenges"]:
                challenge_description = self.chat.prompt(
                    "Describe the challenge in this room. The challenge should be something that the player must overcome to progress. Do not use the term challenge. Describe it in a few sentences, starting with the challenge."
                )
                print("Challenge description:", challenge_description)
                descriptions[key]["challenge"] = challenge_description
                descriptions[key]["text"] += f"\n\n{challenge_description}"
            elif descriptions[key]["is_unusual"]:
                unusual_description = self.chat.prompt(
                    "Describe what makes this room unusual in more detail. This room should be different from the others in some way. Describe it in a few sentences, starting with the unusual aspect."
                )
                print("Unusual description:", unusual_description)
                descriptions[key]["unusual"] = unusual_description
                descriptions[key]["text"] += f"\n\n{unusual_description}"

            descriptions[key]["actions"] = {}
            for n in next_nodes:
                action_n = self.chat.prompt(
                    f'Describe how to get to {n} from {node} in one short imperative sentence of < 8 words, without saying either {n} or {node}, e.g. "Climb the stairs.":',
                    verbose=False,
                )
                print(" - ", action_n, "to", n)
                descriptions[key]["actions"][f"{n[0]}_{n[1]}"] = action_n

            # Generate the image prompt
            image_prompt = self.image_prompt.format(
                location=location, description=description
            )
            image_description = self.image_promt_generator.prompt(
                image_prompt, verbose=False
            )
            print("Image description:", image_description)
            descriptions[key]["image"] = image_description

            descriptions[key]["image_filename"] = f"{node[0]}_{node[1]}.png"

        # Create folder based on location name
        folder_name = location.replace(" ", "_").lower()
        os.makedirs(folder_name, exist_ok=True)

        # Create yaml dump of the descriptions
        with open(os.path.join(folder_name, "descriptions.yaml"), "w") as f:
            OmegaConf.save(descriptions, f)

        # Copy index.html and labyrinth.js to the folder
        shutil.copy(get_index_html_path(), folder_name)
        shutil.copy(get_labyrinth_js_path(), folder_name)

        # Generate images for each room
        for node, description in descriptions.items():
            if node == "world":
                continue
            image_filename = os.path.join(folder_name, description["image_filename"])
            self.generate_image(image_style, description["image"], image_filename)

    def generate_image(
        self, image_style: str, image_description: str, image_filename: str
    ) -> None:
        """Generate an image based on the given description.

        Args:
            image_style: The style of the image to generate.
            image_description: The description of the image to generate.
            image_filename: The filename to save the image to.
        """

        # Generate the image
        image = self.image_generator.generate(image_style + " " + image_description)
        image.save(image_filename)

        print(f"Saved image to {image_filename}")


@hydra.main(version_base=None, config_path="config", config_name="labyrinth")
def main(cfg: DictConfig):
    print("Application Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Create a LabyrinthGenerator
    labyrinth_generator = LabyrinthGenerator(cfg)

    # Generate the default labyrinth
    # labyrinth_generator.generate()
    for world in cfg.worlds:
        print("Generating world:", world)
        world = cfg.worlds[world]
        labyrinth_generator.generate(
            location=world.location,
            goal=world.goal,
            writing_style=world.writing_style,
            image_style=world.image_style,
        )


if __name__ == "__main__":
    main()
