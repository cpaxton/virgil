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

# (c) 2024 by Chris Paxton

# (c) 2024 by Chris Paxton
# Quiz Maker, advanced version

import os
from datetime import datetime
import yaml
import click

from virgil.backend import get_backend, Backend
from virgil.chat import ChatWrapper
from virgil.quiz.parser import ResultParser, QuestionParser


userprompt = """
Enter something to make a quiz about. For example:
- What kind of sandwich are you?
- Which Lord of the Rings character are you?
- Which faction from Iain Banks' Culture series are you?
- Which kitchen utensil are you?

It should be a question.

Go ahead:
"""

prompt_answers = """You are generating a fun, clever Buzzfeed-style personality quiz titled, "{topic}".

There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description.

Results are given with a title and a description, as well as a longer description (1-2 paragraphs) and a prompt for an image generator. You must always end each result with "END RESULT". Image descriptions should be concise and evocative, and should be detailed enough for an image generator to create a good image. Don't use too many names or specific references, but instead focus on the general theme of the result, unless it's a well-known pop culture reference or a famous character.

For example:

Topic: What kind of sandwich are you?
Mostly A's:
Result: Ham and Cheese Sandwich.
Description: You are the classic sandwich, reliable and comforting. You are the go-to, the classic, the reliable. You are comforting and familiar, and people know they can count on you to be there when they need you. Your weakness is that you might be a bit TOO familiar, and people might take you for granted. But you are always there, and you are always delicious.
Image: A sandwich with ham and cheese. The bread is golden brown, and the cheese is melted. There is a bit of lettuce peeking out from the side, and a slice of tomato. The sandwich is cut in half, and you can see the layers of ham and cheese inside.
END RESULT

NEVER repeat a result. Each result will be unique and as specific as possible, and they will be the most popular or obvious things related to the question: {topic}

Every result needs a detailed image description.

After Result C, options will get steadily more unhinged and nonsensical. When prompted, with "Result X", you will generate only the text for that result and no more. End each question with "END RESULT". Provide no other output.

Content will be formatted exactly as, with no extra fields. You will not return or say anything else. Do not use markdown or any other formatting, other than what is above.

You will return:

Topic: (the title of the quiz)
Mostly (letter)'s:
Result: (the result)
Description: (the description)
Image: (A detailed prompt for an image generator)
END RESULT

Topic: {topic}
Mostly A's:
"""

prompt_questions = """
You are generating a fun, clever Buzzfeed-style personality quiz titled, "{topic}".

There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description. All questions will be related to "{topic}", but will get increasingly weird as the quiz goes on.

For example, Question 1 will be weirdness level 1, but Question 3 will be weirdness level 9, and may be something very absurd or personal. It will refer back to the answers of the previous question.

You will also give a detailed prompt for an image generator associated with each question. You will be very clear about the image, and will include sufficient detail. You will not hallucinate.

For example:
Question 1:
Question: You go to your kitchen in the middle of the night. What do you do?
Image: A picture of a kitchen. There is a window, and the moon is shining through. There is a shadow on the wall. The kitchen is dark and quiet, but there's a bit of light coming from the fridge, and a pool of light nearby.
A. Get a glass of water.
B. Make a sandwich.
C. Eat a spoonful of peanut butter.
D. Stare into the fridge.
E. You forgot your phone, grab it and go back to bed.
END QUESTION

If the user answers mostly each letter, the personality quiz will give this as a result:
A: {result_a}: {description_a}
B: {result_b}: {description_b}
C: {result_c}: {description_c}
D: {result_d}: {description_d}
E: {result_e}: {description_e}
END QUESTION

Make sure answers are on-theme. For example, answer A should be something relevant to {result_a}; answer B should be something relevant to {result_b}, etc.

All questions are related to the topic. Even numbered questions are highly related to the topic. For example, if the topic was "What kind of sandwich are you?", a good question might be "What is your favorite condiment?". Odd numbered questions are more unhinged and nonsensical.

NEVER repeat a question. Answers should be different and unique. Never repeat an answer.

After question 3, questions will get steadily more unhinged and nonsensical, as well as increasinly personal. For example, if the topic was "What kind of sandwich are you?", a good question might be "When was the last time you cried while eating a sandwich?".

Question N:
Question: (text of the question)
Image: (a detailed prompt for an image generator)
A. (text of option A - relevant to {result_a})
B. (text of option B - relevant to {result_b})
C. (text of option C - relevant to {result_c})
D. (text of option D - relevant to {result_d})
E. (text of option E - relevant to {result_e})
END QUESTION

When prompted, with "Question N", you will generate only the text and image prompt for that question and no more. End each question with "END QUESTION". Provide no other output. Content will be formatted exactly as above, with no extra fields. You will not return or say anything else.

Topic: {topic}
Question 1:
"""


def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


def generate_quiz(topic: str, backend: Backend, save_with_date: bool = False) -> None:
    chat = ChatWrapper(backend)
    result_parser = ResultParser(chat)
    question_parser = QuestionParser(chat)

    # print(userprompt)
    # topic = input("Enter the title of your dumb personality quiz: ")
    # topic = "Which Lord of the Rings character are you?"
    # topic = "Which faction from Iain Banks' Culture series are you?"
    # topic = "Which kitchen utensil are you?"
    # topic = "What sea creature are you?"
    # topic = "What houseplant are you?"
    # topic = "What kind of sandwich are you?"

    if save_with_date:
        # Add subfolder with datetime for current date and time
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        dirname = os.path.join(date, topic)
        os.makedirs(dirname, exist_ok=True)
    else:
        dirname = topic
        os.makedirs(dirname, exist_ok=True)

    msg = prompt_answers.format(topic=topic)
    res_a = result_parser.prompt(msg=msg, topic=topic, letter="A")
    res_b = result_parser.prompt(topic=topic, letter="B")
    res_c = result_parser.prompt(topic=topic, letter="C")
    res_d = result_parser.prompt(topic=topic, letter="D")
    res_e = result_parser.prompt(topic=topic, letter="E")

    # Save all the results out as a YAML file
    with open(os.path.join(dirname, "results.yaml"), "w") as f:
        yaml.dump({"A": res_a, "B": res_b, "C": res_c, "D": res_d, "E": res_e}, f)

    chat.clear()

    msg = prompt_questions.format(
        num_questions=10,
        topic=topic,
        result_a=res_a["result"],
        result_b=res_b["result"],
        result_c=res_c["result"],
        result_d=res_d["result"],
        result_e=res_e["result"],
        description_a=res_a["description"],
        description_b=res_b["description"],
        description_c=res_c["description"],
        description_d=res_d["description"],
        description_e=res_e["description"],
    )
    q1 = question_parser.prompt(topic=topic, question=1, msg=msg)
    questions = [q1]
    for i in range(2, 11):
        if i % 2 == 0:
            msg = f"Topic: {topic}\nQuestion {i}:"
        else:
            msg = f"Topic: {topic}\nQuestion {i}:"
        q = question_parser.prompt(topic=topic, question=i, msg=msg)
        questions.append(q)

    # Save all the questions out as a YAML file
    with open(os.path.join(dirname, "questions.yaml"), "w") as f:
        yaml.dump(questions, f, allow_unicode=True, sort_keys=False)

    chat.clear()

    # create_images_for_folder(os.path.join(date, topic))


@click.command()
@click.option("--topic", default="", help="The topic of the quiz to generate.")
@click.option(
    "--backend",
    default="gemma-3-12b-it",
    help="The backend to use for generating the quiz.",
)
def main(topic: str = "", backend: str = "gemma-3-12b-it") -> None:
    backend = get_backend(backend)

    # If you specified a quiz...
    if len(topic) > 0:
        generate_quiz(topic, backend)
        return

    # If you did not specify a quiz...
    topics = [
        "Which Lord of the Rings character are you?",
        "Which faction from Iain Banks' Culture series are you?",
        "Which kitchen utensil are you?",
        "What sea creature are you?",
        "What houseplant are you?",
        "What kind of sandwich are you?",
        "What D&D character class are you?",
        "Which programming language are you?",
        "What kind of cat are you?",
        "What kind of dog are you?",
        "What kind of bird are you?",
        "What kind of fish are you?",
        "What kind of reptile are you?",
        "What kind of amphibian are you?",
        "What kind of insect are you?",
        "What kind of arachnid are you?",
        "What kind of mollusk are you?",
        "What kind of crustacean are you?",
        "What kind of arthropod are you?",
        "What kind of worm are you?",
        "What kind of fungus are you?",
        "What kind of bacteria are you?",
        "What kind of virus are you?",
        "What kind of protist are you?",
        "What kind of plant are you?",
        "What kind of tree are you?",
        "What kind of flower are you?",
        "What kind of fruit are you?",
        "What kind of vegetable are you?",
        "What kind of herb are you?",
        "What kind of spice are you?",
        "What kind of condiment are you?",
        "What kind of sauce are you?",
        "What kind of soup are you?",
        "What kind of salad are you?",
        "What kind of appetizer are you?",
        "What kind of entree are you?",
        "What kind of dessert are you?",
        "What kind of drink are you?",
        "What kind of cocktail are you?",
        "What kind of beer are you?",
        "What kind of wine are you?",
        "What kind of spirit are you?",
        "What kind of non-alcoholic beverage are you?",
        "What kind of juice are you?",
        "What kind of soda are you?",
        "What kind of tooth are you?",
        "Which bone are you?",
        "What halloween costume are you?",
        "What halloween creature are you?",
        "Which day in October are you?",
        "What halloween candy are you?",
        # Gemma failed to generate a quiz for "what halloween creature are you?"
        "Which cosmic horror are you devoted to?",
        "To which of the elder gods should you pray?",
        "Which afterlife will you end up in?",
        "Which kind of undead monstrosity will you be?",
        "What holiday are you?",
        "What kind of door are you?",
        "What extremely specific door are you?",
        "What kind of potato are you?",
        "What extremely specific odor are you?",
        "What popular internet meme are you?",
        "What Andy are you?",
        "What quiz are you?",
        "How drunk are you right now?",
        "How did you get so drunk?",
        "Who am I and how did I get here?",
        "What should I name my dog?",
        "How do I get out of here?",
        "What kind of tea are you?",
        "What kind of coffee are you?",
        "What kind of milk are you?",
        "What kind of water are you?",
        "What kind of ice cream are you?",
        "What kind of candy are you?",
        "What kind of chocolate are you?",
        "What kind of snack are you?",
        "What kind of chip are you?",
        "What kind of cracker are you?",
        "What kind of cookie are you?",
        "What kind of cake are you?",
        "What kind of pie are you?",
        "What kind of bread are you?",
        "What kind of pasta are you?",
        "What kind of rice are you?",
        "What kind of grain are you?",
        "What kind of legume are you?",
        "What kind of nut are you?",
        "What kind of seed are you?",
        "What kind of oil are you?",
        "What kind of vinegar are you?",
        # Started from when I was using Gemma3-14b, may 28, 2025
        "Which character from the Stormlight Archive are you?",
    ]

    for topic in topics:
        generate_quiz(topic, backend)

    print("All quizzes generated successfully!")


if __name__ == "__main__":
    main()
