from abc import ABC, abstractmethod


class Social(ABC):
    def __init__(self, username: str):
        self.username = username

    @abstractmethod
    def connect(self):
        """
        Connect to the social media platform.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def post(self, content: str):
        """
        Post content to the social media platform.

        Args:
            content (str): The content to post.
        """
        pass
