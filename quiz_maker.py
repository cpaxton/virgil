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
You are generating a weird personality quiz about {topic}.

For each quiz, you will generate {num_questions} questions. There will be 5 multiple-choice options per question: A, B, C, D, and E. At the end, you will also provide a categorization: if the quiz taker chose mostly A, for example, you will describe what A is, and give a description.

For example, if the topic is "What kind of sandwich are you?" and the first question is "What is your favorite color?" the options might be:
A. Red
B. Blue
C. Green
D. Yellow
E. Black

The quiz is about {topic}.

Question 1:
"""

conversation_history = []

def prompt_llm(msg, is_first_message: bool = False):
    print()

    if is_first_message:
        is_first_message = False
        full_msg = prompt.format(num_questions=10, topic=topic)

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
    print(f"Parse time taken: {parse_dt} seconds")
    print(f"Story time taken: {t1-t0:.2f} seconds")
    print("----------------")

print(userprompt)
print()
is_first_message = True

topic = input("Say or do something (empty to quit):")
prompt_llm(topic, is_first_message=True)
prompt_llm("Question 2:")
