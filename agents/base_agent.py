from abc import ABC, abstractmethod

class BaseAgent(ABC):

    def __init__(self, name="Agent"):
        self.name = name

    @abstractmethod
    def choose_action(self, game) -> int:
        """
        Given the current game state, return a column number.
        """
        pass