# (c) 2024 by Chris Paxton
# This is the proof of concept for the simple quiz maker. It uses an LLM to generate a quiz based on a user's input.

import torch
from transformers import pipeline
import timeit
from termcolor import colored

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)

import sys

userprompt = """
Enter something to make a quiz about.
"""

prompt = """
You are generating a weird personality quiz titled, "{topic}".

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

End each question with "END QUESTION". Provide no other output.

The quiz is about {topic}.

Topic: {topic}
Question 1:
"""

conversation_history = []

def prompt_llm(msg, is_first_message: bool = False):
    print()

    if is_first_message:
        is_first_message = False
        full_msg = prompt.format(num_questions=10, topic=topic)
    else:
        full_msg = msg

    conversation_history.append({"role": "user", "content": full_msg})

    messages = conversation_history.copy()
    t0 = timeit.default_timer()
    outputs = pipe(messages, max_new_tokens=256)
    t1 = timeit.default_timer()
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Print the assistant's response
    print(assistant_response)
    print("----------------")
    print(f"Generator time taken: {t1-t0:.2f} seconds")
    print("----------------")

print(userprompt)
print()
is_first_message = True

topic = input("Enter the title of your dumb personality quiz: ")
prompt_llm(topic, is_first_message=True)
for i in range(1,5):
    prompt_llm(f"Topic: {topic}\nQuestion {i+1}:")

prompt_llm("""If you were mostly A, you were:""")
prompt_llm("""If you were mostly B, you were:""")
prompt_llm("""If you were mostly C, you were:""")
prompt_llm("""If you were mostly D, you were:""")
prompt_llm("""If you were mostly E, you were:""")
