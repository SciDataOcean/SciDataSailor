from abc import ABC, abstractmethod


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def trigger_tag(self) -> str:
        """Tag name used in <xxx> </xxx> pairs."""
        pass

    @abstractmethod
    async def execute(self, content: str, **kwargs) -> str:
        """Execute the tool and return the result string."""
        pass
