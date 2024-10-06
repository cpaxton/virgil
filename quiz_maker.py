# (c) 2024 by Chris Paxton
# Quiz Maker, advanced version

import timeit
from termcolor import colored
import datetime

from virgil.backend import Gemma
backend = Gemma()

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
Description: You are a classic sandwich, reliable and comforting. You are a ham and cheese sandwich.
Long Description: You are the go-to, the classic, the reliable. When people think of a sandwich, they think of you. You are a ham and cheese sandwich, the classic combination of salty ham and creamy cheese. You are comforting and familiar, and people know they can count on you to be there when they need you.
Image: a sandwich with ham and cheese.
END RESULT

After Result C, options will get steadily more unhinged and nonsensical. When prompted, with "Result X", you will generate only the text for that result and no more. End each question with "END RESULT". Provide no other output.

Content will be formatted exactly as, with no extra fields. You will not return or say anything else. You will return:

Topic: (the title of the quiz)
Mostly (letter)'s:
Result: (the result)
Description: (the description)
Long Description: (the long description, 1-2 paragraphs)

Topic: {topic}
Mostly A's:
"""

prompt_questions = """
You are generating a dumb, weird, BuzzFeed-style personality quiz titled, "{topic}".

There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description.

You will also give a prompt for an image generator associated with each question.

For example:

Topic: What kind of sandwich are you?
Question 1:
Question: What is your favorite color?
Image: a sandwich with a red tomato, a green lettuce leaf, and a yellow cheese slice.
A. Red
B. Blue
C. Green
D. Yellow
E. Black
END QUESTION

After question 3, questions will get steadily more unhinged and nonsensical. When prompted, with "Question N", you will generate only the text for that question and no more. End each question with "END QUESTION". Provide no other output.

Content will be formatted exactly as above, with no extra fields. You will not return or say anything else.

Topic: {topic}
Question 1:
"""

conversation_history = []

def prompt_llm(full_msg):
    print()

    conversation_history.append({"role": "user", "content": full_msg})

    messages = conversation_history.copy()
    t0 = timeit.default_timer()
    outputs = backend(messages, max_new_tokens=256)
    t1 = timeit.default_timer()
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Print the assistant's response
    print()
    print(colored("User prompt:\n", "green") + full_msg)
    print()
    print(colored("Response:\n", "blue") + assistant_response)
    print("----------------")
    print(f"Generator time taken: {t1-t0:.2f} seconds")
    print("----------------")

print(userprompt)
print()

# topic = input("Enter the title of your dumb personality quiz: ")
topic = "Which Lord of the Rings character are you?"

msg = prompt_answers.format(topic=topic)
prompt_llm(msg)
prompt_llm(f"Topic: {topic}\nMostly B's:")
prompt_llm(f"Topic: {topic}\nMostly C's:")
prompt_llm(f"Topic: {topic}\nMostly D's:")
prompt_llm(f"Topic: {topic}\nMostly E's:")

msg = prompt_questions.format(num_questions=10, topic=topic)
prompt_llm(msg)
prompt_llm(f"Topic: {topic}\nQuestion 2:")
prompt_llm(f"Topic: {topic}\nQuestion 3:")
prompt_llm(f"Topic: {topic}\nQuestion 4:")
prompt_llm(f"Topic: {topic}\nQuestion 5:")

