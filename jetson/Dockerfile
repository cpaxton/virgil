FROM dustynv/l4t-text-generation:r35.3.1

# pip3 install --upgrade transformers && pip3 install --user termcolor diffusers discord.py && export PYTHONPATH=/virgil:$PYTHONPATH && python3 -m virgil.friend.friend"
RUN  pip3 install --upgrade transformers && pip3 install --user termcolor diffusers discord.py && export PYTHONPATH=/virgil:$PYTHONPATH

# Entry point is friend
CMD ["python3", "-m", "virgil.friend.friend"]
