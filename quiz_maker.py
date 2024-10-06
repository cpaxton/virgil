# (c) 2024 by Chris Paxton
# Quiz Maker, advanced version

import os
from datetime import datetime
import yaml

from virgil.backend import Gemma
from virgil.chat import ChatWrapper
from virgil.quiz.parser import ResultParser, QuestionParser
from virgil.quiz.create_images import create_images_for_folder

userprompt = """
Enter something to make a quiz about. For example:
- What kind of sandwich are you?
- Which Lord of the Rings character are you?
- Which faction from Iain Banks' Culture series are you?
- Which kitchen utensil are you?

It should be a question.

Go ahead:
"""

prompt_answers = """You are generating a weird personality quiz titled, "{topic}".

There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description.

Results are given with a title and a description, as well as a longer description (1-2 paragraphs) and a prompt for an image generator.

For example:

Topic: What kind of sandwich are you?
Mostly A's:
Result: Ham and Cheese Sandwich.
Description: You are the classic sandwich, reliable and comforting. You are the go-to, the classic, the reliable. You are comforting and familiar, and people know they can count on you to be there when they need you. Your weakness is that you might be a bit TOO familiar, and people might take you for granted. But you are always there, and you are always delicious.
Image: A picture of a sandwich with ham and cheese. The bread is golden brown, and the cheese is melted. There is a bit of lettuce peeking out from the side, and a slice of tomato. The sandwich is cut in half, and you can see the layers of ham and cheese inside.
END RESULT

NEVER repeat a result. Each result will be unique, and they will be the most popular or obvious things related to the question: {topic}

Every result needs a detailed image description as well.

After Result C, options will get steadily more unhinged and nonsensical. When prompted, with "Result X", you will generate only the text for that result and no more. End each question with "END RESULT". Provide no other output.

Content will be formatted exactly as, with no extra fields. You will not return or say anything else. Do not use markdown or any other formatting, other than what is above.

You will return:

Topic: (the title of the quiz)
Mostly (letter)'s:
Result: (the result)
Description: (the description)
Image: A picture of (a detailed prompt for an image generator)

Topic: {topic}
Mostly A's:
"""

prompt_questions = """
You are generating a dumb, weird, BuzzFeed-style personality quiz titled, "{topic}".

There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description. All questions will be related to "{topic}", but will get increasingly weird as the quiz goes on.

For example, Question 1 will be weirdness level 1, but Question 3 will be weirdness level 9, and may be something very absurd or personal. It will refer back to the answers of the previous question.

You will also give a detailed prompt for an image generator associated with each question. You will be very clear about the image, and willl include sufficient detail. You will not hallucinate.

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
A: {result_a}
B: {result_b}
C: {result_c}
D: {result_d}
E: {result_e}

So make sure answers are on-theme. For example, answer A should be something relevant to {result_a}; answer B should be something relevant to {result_b}, etc.

Even numbered questions are highly related to the topic. For example, if the topic was "What kind of sandwich are you?", a good question might be "What is your favorite condiment?". Odd numbered questions are more unhinged and nonsensical.

NEVER repeat a question. Answers should be different and unique.

After question 3, questions will get steadily more unhinged and nonsensical, as well as increasinly personal. For example, if the topic was "What kind of sandwich are you?", a good question might be "When was the last time you cried while eating a sandwich?".

Question N:
Question: (text of the question)
Image: A picture of (a detailed prompt for an image generator)
A. (text of option A)
B. (text of option B)
C. (text of option C)
D. (text of option D)
E. (text of option E)
END QUESTION

When prompted, with "Question N", you will generate only the text for that question and no more. End each question with "END QUESTION". Provide no other output. Content will be formatted exactly as above, with no extra fields. You will not return or say anything else.

Topic: {topic}
Question 1:
"""

def generate_quiz(topic: str, backend: Gemma) -> None:
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

    # Add subfolder with datetime for current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    os.makedirs(os.path.join(date, topic), exist_ok=True)

    msg = prompt_answers.format(topic=topic)
    res_a = result_parser.prompt(msg=msg, topic=topic, letter="A")
    res_b = result_parser.prompt(topic=topic, letter="B")
    res_c = result_parser.prompt(topic=topic, letter="C")
    res_d = result_parser.prompt(topic=topic, letter="D")
    res_e = result_parser.prompt(topic=topic, letter="E")

    # Save all the results out as a YAML file
    with open(os.path.join(date, topic, "results.yaml"), "w") as f:
        yaml.dump({"A": res_a, "B": res_b, "C": res_c, "D": res_d, "E": res_e}, f)

    chat.clear()

    msg = prompt_questions.format(num_questions=10, topic=topic, result_a=res_a["result"], result_b=res_b["result"], result_c=res_c["result"], result_d=res_d["result"], result_e=res_e["result"])
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
    with open(os.path.join(date, topic, "questions.yaml"), "w") as f:
        yaml.dump(questions, f)

    chat.clear()

    # create_images_for_folder(os.path.join(date, topic)) 

def main():
    # The first set of topics
    topics = ["Which Lord of the Rings character are you?", "Which faction from Iain Banks' Culture series are you?", "Which kitchen utensil are you?", "What sea creature are you?", "What houseplant are you?", "What kind of sandwich are you?", "What D&D character class are you?", "Which programming language are you?", "What kind of cat are you?", "What kind of dog are you?", "What kind of bird are you?", "What kind of fish are you?", "What kind of reptile are you?", "What kind of amphibian are you?", "What kind of insect are you?", "What kind of arachnid are you?", "What kind of mollusk are you?", "What kind of crustacean are you?", "What kind of arthropod are you?", "What kind of worm are you?", "What kind of fungus are you?", "What kind of bacteria are you?"]
    # Yet more topics
    topics2 = ["What kind of virus are you?", "What kind of protist are you?", "What kind of plant are you?", "What kind of tree are you?", "What kind of flower are you?", "What kind of fruit are you?", "What kind of vegetable are you?", "What kind of herb are you?", "What kind of spice are you?", "What kind of condiment are you?", "What kind of sauce are you?", "What kind of soup are you?", "What kind of salad are you?", "What kind of appetizer are you?", "What kind of entree are you?", "What kind of dessert are you?", "What kind of drink are you?", "What kind of cocktail are you?", "What kind of beer are you?", "What kind of wine are you?", "What kind of spirit are you?", "What kind of non-alcoholic beverage are you?", "What kind of juice are you?", "What kind of soda are you?", "What kind of tea are you?", "What kind of coffee are you?", "What kind of milk are you?", "What kind of water are you?", "What kind of ice cream are you?", "What kind of candy are you?", "What kind of chocolate are you?", "What kind of snack are you?", "What kind of chip are you?", "What kind of cracker are you?", "What kind of cookie are you?", "What kind of cake are you?", "What kind of pie are you?", "What kind of bread are you?", "What kind of pasta are you?", "What kind of rice are you?", "What kind of grain are you?", "What kind of legume are you?", "What kind of nut are you?", "What kind of seed are you?", "What kind of oil are you?", "What kind of vinegar are you?"]
    backend = Gemma()
    for topic in topics:
        generate_quiz(topic, backend)

if __name__ == "__main__":
    main()
