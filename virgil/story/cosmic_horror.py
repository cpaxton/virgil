import timeit
from termcolor import colored

from virgil.backend import get_backend

import sys

userprompt = """Your sister called you at 3am three days ago, drunk and angry and scared as shit. She said was going home. But you'd both never sworn to go back there again, back to that awful, haunting town on the water. It'd be better if the water rose and that whole awful place washed away, that's what you both used to say. But she said she remembered what happened, all those years ago, and that she needed to, and then she hung up.

You haven't heard from her since. Your phone calls go to voicemail, and so you packed your bags, bailed on your classes, and got on a last-minute grayhound bus from Columbus. You'll have to deal with the follout later.

You got off the Grayhound a few minutes ago; you're dead tired but you can't sleep yet. No money for a rental, so your only way to your old family home is via the #10 city bus line. Now the beat up old bus stop stands before you, rusted and twisted by years in this awful town. What do you do?"""

prompt = f"""
You are an AI storyteller running an interactive game in a cosmic horror story setting, inspired by Lovecraft. I will be interacting with you and you are to give me details about the world and the consequences of the actions. Keep answers short.

The setting is this: we are at a bus stop outside an old, declining rust belt city. The bus stop is populated by a single old man, named Silas, who has graying hair and a worn peacoat. He's got an old pipe, like something out of a 1920s silent film, and he starts packing it after he sits down.

Information that you may use but are not to bring up unless it is relevant:
    - the city is called Greenwood
    - You are in Ohio
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

My actions will be in the form action:description. For example, "look at the bus stop" or "talk to Silas".

After an action, you will elaborate on what I did. For example, if the action was "look at the bus stop", you might say "You look around the bus stop. It's a beat up old thing, rusted and twisted by years in this awful town."

When continuing, you will be concise.

Once the bus arrives or the conversation ends, just say END in all caps.

From here on, stay in character. You will answer concisely:

First, state what I do - elaborate on it for 1-2 sentences. Then, state 1-2 sentences of what I see, or do, or how the relevant characters in teh wrold react. For example:

Input: "look at the bus stop"
Output:
You look around at the bus stop. It's a beat up old thing, rusted, but there's a good seat, and you can see a diner across the road. There's an old man with a pipe sitting on the bench, watching you.

Input: "talk to Silas"
Output:
You walk over to Silas and strike up a conversation. Silas looks up from his pipe and gives you a nod. "You're a college student, right? You look like you've been up all night."

Input: "ask about me"
Output:
You ask Silas about yourself. Silas looks at you, then looks away. "You look like you've been up all night, kid," he says. "You should get some sleep. What's a college student like you doing up here in Greenwood, anyway? Anyone with a future left a long time ago."

Input: "ask about sister"
Output:
You ask Silas about your sister. Silas looks at you, then looks away. "I haven't seen her," he says. "But a lot of folks have gone missing lately. You should be careful."

Input: "ask about disappearances"
Output:
You ask about the disappearances, but Silas looks away, uncomfortable. "I don't want to talk about it," he says. "Just keep your curtains drawn at night, and keep your head down."
"""

prompt2 = """
You are parsing commands or instructions into three parts: an action, a target, and any remaining text joining the two. For example:

Input: "I look around the bus stop"
output:
action=look
target=bus stop
joiner=at the 

Input: "sit down"
action=sit
target=bus stop
joiner=at the

Input: "take a seat"
output:
action=take a seat
target=bus stop
joiner=at the

input: "look around"
output:
action=look
target=bus stop
joiner=at the

Input: "wait, what about the lights?"
output:
action=ask
target=lights
joiner=about the

Input: "talk to Silas"
output:
action=talk
target=Silas
joiner=to

Input: "ask Silas about the lights"
output:
action=ask
target=lights
joiner=about the

Input: "talk to Silas about the disappearances"
output:
action=talk
target=disappearances
joiner=about the

Input: "use the phone"
output:
action=use
target=phone
joiner=the

Actions should only be one word, if possible.

You will always provide all three terms; infer the joiner if necessary.
Give only the output, no other information, for the next input.

Input: "I agree with Silas."
Output:
action=agree
target=Silas
joiner=with

Input: 
"""

conversation_history = []
backend = get_backend("gemma")
verbose = False

def parse(msg) -> tuple[str, str, str]:
    """Parse the user's message into an action, a target, and a joiner.

    Args:
        msg (str): The user's message.

    Returns:
        tuple[str, str, str]: The action, target, and joining text.
    """
    t0 = timeit.default_timer()
    new_message = {"role": "user", "content": prompt2 + msg}
    messages = [new_message]

    outputs = backend(messages, max_new_tokens=256)
    t1 = timeit.default_timer()
    msg = outputs[0]["generated_text"][-1]["content"].strip()

    # Read each line
    action = ""
    description = ""
    joiner = ""
    for line in msg.split("\n"):
        if line.startswith("action="):
            action = line.split("=")[1]
        elif line.startswith("target="):
            description = line.split("=")[1]
        elif line.startswith("joiner="):
            joiner = line.split("=")[1]
    return action, description, joiner, t1-t0

print(userprompt)
print()
verbose = False
is_first_message = True
while True:
    print()
    print()
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
    else:
        full_msg = parsed_msg

    # conversation_history.append(parsed_msg)
    # conversation_history.append({"role": "user", "content": parsed_msg})
    conversation_history.append({"role": "user", "content": full_msg})
    # Prepare the messages including the conversation history
    messages = conversation_history.copy()
    t0 = timeit.default_timer()

    outputs = backend(messages, max_new_tokens=256)
    t1 = timeit.default_timer()
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Print the assistant's response
    print(assistant_response)
    if "END" in assistant_response:
        break
    if verbose:
        print("----------------")
        print(f"Parse time taken: {parse_dt} seconds")
        print(f"Story time taken: {t1-t0:.2f} seconds")
        print("----------------")
