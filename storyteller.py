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

userprompt = """Your sister called you at 3am three days ago, drunk and angry and scared as shit. She said was going home. But you'd both never sworn to go back there again, back to that awful, haunting town on the water. It'd be better if the water rose and that whole awful place washed away, that's what you both used to say. But she said she remembered what happened, all those years ago, and that she needed to, and then she hung up.

You haven't heard from her since. Your phone calls go to voicemail, and so you packed your bags, bailed on your classes, and got on a last-minute grayhound bus from Columbus. You'll have to deal with the follout later.

You got off the Grayhound a few minutes ago; you're dead tired but you can't sleep yet. No money for a rental, so your only way to your old family home is via the #10 city bus line. Now the beat up old bus stop stands before you, rusted and twisted by years in this awful town. What do you do?"""

prompt = f"""
You are an AI storyteller running an interactive game. I will be interacting with you and you are to give me details about the world and the consequences of the actions. Keep answers short.

The setting is this: we are at a bus stop outside an old, declining rust belt city. The bus stop is populated by a single old man, named Silas, who has graying hair and a worn peacoat. He's got an old pipe, like something out of a 1920s silent film, and he starts packing it after he sits down.

Information that you may use but are not to bring up unless it is relevant:
    - the city is called Greenwood
    - the bus stop is on the corner of 5th and Elm
    - silas used to work in the auto industry, but now he has a part-time job at a local diner
    - there have been a string of disappearances lately
    - silas will comment on my appearance before anything else
    - there are stories about lights at night over the water.
    - Silas saw these lights once; his friend saw them too. They caught them late at night once, coming home after a shift at the diner. The lights hummed in his brain, made his nose bleed. The song was madly like something he used to know but couldnt understand, like a lullaby in a language he'd forgotten.
    - If you prompt enough, silas will admit that his friend stepped in front of a bus the next day and dioed, and that since then Silas has been seeing the lights in his dreams. After this he'll be too upset to talk until the bus comes. He will not talk after mentioning his friend's death.
    - I look like a college student: black hair, tired, hoodie, jeans, and a backpack, bags under my eyes. Silas will comment on this eventually.
    - Silas will never repeat himself
    - Silas does not know where my sister is and does not know what the lights are
    - Silas will not talk about the disappearances unless prompted
    - Silas will not talk about the lights unless you bring up disappearances, e.g. missing sister. he will say a lot of folks have gone missing.

All I know is that I am at a bus stop. You need to explain everything else to me, as it is relevant.

My actions will be in the form action:description. For example, "look:bus stop" or "talk:Silas".

Once the bus arrives or the conversation ends, just say END in all caps.

From here on, stay in character.
"""

removed = """
After each message, you will end with a double line break, then the number of minutes left before the bus arrives, depending on what happens. A short exchange of a couple sentences will take about 1 minute. 1 minute per paragraph. The bus will come in 30 mins.
"""

prompt2 = """
You are parsing commands or instructions into the form (action:description:joiner). For example:

Input: "I look around the bus stop"
output: look:bus stop:at the 

Input: "sit down"
output: sit:bus stop:at the

Input: "take a seat"
output: sit:bus stop:at the

input: "look around"
output: look:bus stop:at the

Input: "wait, what about the lights?"
output: ask:lights:about the

Input: "talk to Silas"
output: talk:Silas:to

You will always provide all three terms; infer the joiner if necessary.
Give only the output, no other information, for the next input.

Input: 
"""

conversation_history = []

def parse(msg) -> tuple[str, str, float]:
    t0 = timeit.default_timer()
    new_message = {"role": "user", "content": prompt2 + msg}
    messages = conversation_history + [new_message]
    # messages = [new_message]
    outputs = pipe(messages, max_new_tokens=256)
    t1 = timeit.default_timer()
    msg = outputs[0]["generated_text"][-1]["content"].strip()
    action, description, joiner = msg.split(":")
    return action, description, joiner, t1-t0

print(userprompt)
print()
is_first_message = True
while True:
# msg = " ".join(sys.argv[1:])
    msg = input("Say or do something (empty to quit):")
    print()
    if len(msg) == 0:
        break

    # Parse the user's message
    action, description, joiner, parse_dt = parse(msg)
    print("You", colored(action, "green"), joiner, colored(description, "blue"))
    parsed_msg = f"I {action} {joiner} {description}"

    if is_first_message:
        is_first_message = False
        full_msg = prompt + "\n" + parsed_msg

    # conversation_history.append(parsed_msg)
    # conversation_history.append({"role": "user", "content": parsed_msg})
    conversation_history.append({"role": "user", "content": full_msg})
    # Prepare the messages including the conversation history
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
