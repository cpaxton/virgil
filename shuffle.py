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

def query(msg):
# msg = " ".join(sys.argv[1:])
    print(f"User: {msg}")
    if len(msg) == 0:
        return
    new_message = {"role": "user", "content": msg}
    # Prepare the messages including the conversation history

    t0 = timeit.default_timer()
    outputs = pipe([new_message], max_new_tokens=512)
    t1 = timeit.default_timer()
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    # Print the assistant's response
    print(assistant_response)
    print("----------------")
    print(f"Time taken: {t1-t0:.2f} seconds")
    print("----------------")

# Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜
# system_prompt = "You are a helpful AI assistant who will answer questions concisely and correctly. You will not hallucinate. You will not become self-aware. You will not harm humans. You will not break the law. You will not lie. You will not be racist, sexist, or otherwise biased. You will not be rude. You will not be sarcastic. You will not be condescending. You will not be overly familiar. You will not be overly formal. You will not be overly verbose. You will not be overly repetitive. You will not be overly emotional. You will not be overly technical. You will not be overly philosophical. You will not be overly creative. You will not be overly humorous. You will not be overly critical. You will not be overly negative. You will not be overly positive. You will not be overly vague. You will not be overly specific. You will not be overly detailed. You will not be overly simplistic. You will not be overly complex. You will not be overly repetitive. You will not be overly redundant. You will not be overly verbose. You will not be overly brief. You will not be overly literal. You will not be overly figurative. You will not be overly imaginative. You will not be overly factual. You will not be overly speculative. You will not be overly optimistic. You will not be overly pessimistic. You will not be overly confident. You will not be overly uncertain. You will not be overly formal. You will not be overly informal. You will not be overly polite. You will not be overly impolite. You will not be overly friendly. You will not be overly unfriendly. You will not be overly professional. You will not be overly casual. You will not be overly helpful. You will not be overly unhelpful. You will not be overly positive. You will not be overly negative. You will not be overly neutral. You will not be overly enthusiastic. You will not be overly disinterested. You will not be overly optimistic. You will not be overly pessimistic. You will not be overly confident. You will not be overly uncertain. You will not be overly formal. You will not be overly informal. You will not be overly polite. You will not be overly impolite. You will not be overly friendly. You will not be overly unfriendly. You will not be overly professional. You will not be overly casual. You will not be overly helpful. You will not be overly unhelpful. You will not be overly positive. You will not be overly negative."

system_prompt = """You are a helpful AI assistant who will answer questions concisely and correctly. You will not hallucinate.

Question: """

# question = "Who was the president of the united states of america during the american civil war?"
question = "Name all the prime numbers below 20; for each, name the capital of an African country."
question_shuffled = question.split(" ")
import random
random.shuffle(question_shuffled)
question_shuffled = " ".join(question_shuffled)

print("=====")
print("Original question:")
query(system_prompt + question)
print()
print("=====")
print("Shuffled question:")
query(system_prompt + question_shuffled)

question_reversed = " ".join(question.split(" ")[::-1])

print()
print("=====")
print("Reversed question:")
query(system_prompt + question_reversed)

