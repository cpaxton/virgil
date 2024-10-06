# # (c) 2024 by Chris Paxton

from virgil.parser import Parser
from typing import Dict, Any, Optional

from termcolor import colored

class ResultParser(Parser):
    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Results are of the format:

            Topic: (the title of the quiz)
            Mostly (letter)'s:
            Result: (the result)
            Description: (the description)
            Image: A picture of (a detailed prompt for an image generator)

        We will process this into a dict, with the following keys:
        - topic
        - letter
        - result
        - description
        - image


        Args:
            text (str): The text to parse.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of the parsed text.
        """
        text = self.prune_to_key(text, "END RESULT")
        # Split the text into lines
        lines = text.split("\n")
        # Initialize the result dictionary
        result = {}
        # Iterate over the lines
        for line in lines:
            # Split the line into key and value
            if len(line.strip()) == 0:
                break
            elif line.startswith("Mostly"):
                key, value = line.split(" ", 1)
                key = "letter"
                value = value[0]
            else:
                key, value = line.split(": ", 1)

            # Add the key and value to the result dictionary
            result[key.lower()] = value
    
        # Verify that all the required keys are present
        required_keys = ["result", "description", "image"]
        if all(key in result for key in required_keys):
            return result
        else:
            print("!!! Missing key in result !!!")
            for key in required_keys:
                if key not in result:
                    print("Missing:", key)
            breakpoint()
            return None

    def prompt(self, topic: str, letter: str, msg: Optional[str] = None) -> str:
        """Prompt the LLM and return the parsed response."""
        if msg is not None:
            res = self.chat.prompt(msg)
        else:
            res = self.chat.prompt(f"Topic: {topic}\nMostly {letter}'s:")
        res = self.parse(res)
        res["letter"] = letter
        res["topic"] = topic

        print()
        print(colored("-" * 10 + str(letter) + "-" * 10, "green"))
        for key, value in res.items():
            print(colored(f"{key}:", "green"), value)

        return res


class QuestionParser(Parser):
    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Questions are of the format:

            Question: (the question)
            Image: A picture of (a detailed prompt for an image generator)
            A. (option A)
            B. (option B)
            C. (option C)
            D. (option D)
            E. (option E)
            END QUESTION

        We will process this into a dict, with the following keys:
        - question
        - image
        - options
            - A
            - B
            - C
            - D
            - E
            etc

        Args:
            text (str): The text to parse.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of the parsed text.
        """

        text = self.prune_to_key(text, "END QUESTION")
        # Split the text into lines
        lines = text.split("\n")
        # Initialize the result dictionary
        result = {}
        # Initialize the options dictionary
        options = {}
        # Iterate over the lines
        for line in lines:
            # Split the line into key and value
            if len(line.strip()) == 0:
                break
            elif line.startswith("Question"):
                key, value = line.split(": ", 1)
                key = "question"
            elif line.startswith("Image"):
                key, value = line.split(": ", 1)
            elif line.startswith("A."):
                options["A"] = line[3:]
            elif line.startswith("B."):
                options["B"] = line[3:]
            elif line.startswith("C."):
                options["C"] = line[3:]
            elif line.startswith("D."):
                options["D"] = line[3:]
            elif line.startswith("E."):
                options["E"] = line[3:]

            # Add the key and value to the result dictionary
            result[key.lower()] = value
        result["options"] = options
        # Verify that all the required keys are present
        required_keys = ["question", "image", "options"]
        if all(key in result for key in required_keys):
            return result
        else:
            breakpoint()
            return None

    def prompt(self, topic: str, question: int, msg: Optional[str] = None) -> str:
        """Prompt the LLM and return the parsed response."""
        if msg is not None:
            res = self.chat.prompt(msg)
        else:
            res = self.chat.prompt(f"Topic: {topic}\nQuestion {question}:")
        res = self.parse(res)

        print()
        print(colored("-" * 10 + str(question) + "-" * 10, "blue"))
        for key, value in res.items():
            print(colored(f"{key}:", "blue"), value)

        return res
