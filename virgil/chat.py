# # (c) 2024 by Chris Paxton

from virgil.backend import Backend


class ChatWrapper:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.conversation_history = []
