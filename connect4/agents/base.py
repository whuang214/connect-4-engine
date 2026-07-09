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

    def reset_stats(self) -> None:
        """
        Reset any internal evaluation/search stats.
        Agents that track stats should override this.
        """
        pass

    def get_stats(self) -> dict:
        """
        Return a dictionary of agent-specific stats.
        Agents that do not track stats can just return {}.
        """
        return {}

    def print_stats(self) -> None:
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print(f"{self.name.upper()} INTERNAL STATS")
        print("=" * 60)

        if not stats:
            print("No internal stats tracked.")
            print("=" * 60)
            return

        for section, values in stats.items():
            print(f"\n{section}:")

            for key, value in values.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}") 