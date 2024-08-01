import torch
from transformers import pipeline
import timeit

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)

import sys

conversation_history = []

while True:
# msg = " ".join(sys.argv[1:])
    msg = input("Enter a message (empty to quit):")
    if len(msg) == 0:
        break
    new_message = {"role": "user", "content": msg}
    conversation_history.append(new_message)
    # Prepare the messages including the conversation history
    messages = conversation_history.copy()
    t0 = timeit.default_timer()
    outputs = pipe(messages, max_new_tokens=512)
    t1 = timeit.default_timer()
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Print the assistant's response
    print(assistant_response)
    print("----------------")
    print(f"Time taken: {t1-t0:.2f} seconds")
    print("----------------")

# Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜

