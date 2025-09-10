from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    Provides a common interface that makes adding new agents simple.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.session_data = {}  # Per-session state
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass
    
    @abstractmethod
    def can_handle(self, message: str, context: Dict = None) -> float:
        """
        Return confidence score (0-1) for handling this message.
        Higher score means more confidence in handling the request.
        """
        pass
    
    @abstractmethod
    async def handle(self, messages: List[BaseMessage], session_id: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process the messages and return response.
        
        Args:
            messages: List of conversation messages
            session_id: Unique session identifier
            context: Additional context data
            
        Returns:
            Dict with 'content' and optionally 'media', 'ui_commands', etc.
        """
        pass
    
    def requires_session_lock(self) -> bool:
        """Whether this agent needs exclusive access (like video stream)"""
        return False
    
    def get_session_timeout(self) -> int:
        """Session timeout in seconds"""
        return 300  # 5 minutes default
    
    def cleanup_session(self, session_id: str):
        """Clean up any session-specific resources"""
        if session_id in self.session_data:
            del self.session_data[session_id]