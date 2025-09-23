from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.messages import BaseMessage

class BaseAgent(ABC):
    """
    Base class for agents. Simplified for direct usage without orchestrator.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def handle(self, messages: list[BaseMessage], session_id: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process the messages and return response.

        Args:
            messages: List of conversation messages
            session_id: Unique session identifier
            context: Additional context data

        Returns:
            Dict with 'content' key containing the response
        """
        pass