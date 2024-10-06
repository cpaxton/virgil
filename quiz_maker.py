# (c) 2024 by Chris Paxton
# Quiz Maker, advanced version

from virgil.backend import Gemma
from virgil.chat import ChatWrapper

userprompt = """
Enter something to make a quiz about.
"""

prompt_answers = """You are generating a weird personality quiz titled, "{topic}".

There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description.

Results are given with a title and a description, as well as a longer description (1-2 paragraphs) and a prompt for an image generator.

For example:

Topic: What kind of sandwich are you?
Mostly A's:
Result: You are a *Ham and cheese* sandwich.
Description: You are a classic sandwich, reliable and comforting. You are a ham and cheese sandwich. You are the go-to, the classic, the reliable. When people think of a sandwich, they think of you: the classic combination of salty ham and creamy cheese. You are comforting and familiar, and people know they can count on you to be there when they need you.
Image: A picture of a sandwich with ham and cheese.
END RESULT

After Result C, options will get steadily more unhinged and nonsensical. When prompted, with "Result X", you will generate only the text for that result and no more. End each question with "END RESULT". Provide no other output.

Content will be formatted exactly as, with no extra fields. You will not return or say anything else. You will return:

Topic: (the title of the quiz)
Mostly (letter)'s:
Result: (the result)
Description: (the description)
Image: (a detailed prompt for an image generator)

Topic: {topic}
Mostly A's:
"""

prompt_questions = """
You are generating a dumb, weird, BuzzFeed-style personality quiz titled, "{topic}".

There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description. All questions will be at least tangentially related to the "{topic}", but will get increasingly weird as the quiz goes on.

For example, Question 1 will be weirdness level 1, but Question 3 will be weirdness level 9, and may be something very absurd or personal. It will refer back to the answers of the previous question.

You will also give a detailed prompt for an image generator associated with each question. You will not hallucinate.

For example:

Question 1:
Question: (text of the question)
Image: (a detailed prompt for an image generator)
A. (text of option A)
B. (text of option B)
C. (text of option C)
D. (text of option D)
E. (text of option E)
END QUESTION

Questions should be somewhat related to the topic, but not too much. For example, if the topic was "What kind of sandwich are you?", a good question might be "What is your favorite condiment?".

After question 3, questions will get steadily more unhinged and nonsensical, as well as increasinly personal. For example, if the topic was "What kind of sandwich are you?", a good question might be "When was the last time you cried while eating a sandwich?".

When prompted, with "Question N", you will generate only the text for that question and no more. End each question with "END QUESTION". Provide no other output.

Content will be formatted exactly as above, with no extra fields. You will not return or say anything else.

Question 1:
"""


def main():
    backend = Gemma()
    chat = ChatWrapper(backend)

    print(userprompt)
    # topic = input("Enter the title of your dumb personality quiz: ")
    topic = "Which Lord of the Rings character are you?"
    # topic = "Which faction from Iain Banks' Culture series are you?"

    msg = prompt_answers.format(topic=topic)
    chat.prompt(msg)
    chat.prompt("Mostly B's:")
    chat.prompt("Mostly C's:")
    chat.prompt("Mostly D's:")
    chat.prompt("Mostly E's:")

    msg = prompt_questions.format(num_questions=10, topic=topic)
    chat.prompt(msg)
    chat.prompt("Question 2:")
    chat.prompt("Question 3:")
    chat.prompt("Question 4:")
    chat.prompt("Question 5:")


if __name__ == "__main__":
    main()
