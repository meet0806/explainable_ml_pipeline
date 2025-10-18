"""
Communication Protocol for Agent Messaging
Defines structured JSON communication between agents
"""

from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import json


class MessageType(str, Enum):
    """Types of messages exchanged between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    INFO = "info"
    DECISION = "decision"


class AgentMessage(BaseModel):
    """
    Structured message format for agent communication
    All agents use this standardized format for interoperability
    """
    sender: str = Field(..., description="Agent name that sent the message")
    receiver: str = Field(..., description="Target agent name")
    message_type: MessageType = Field(..., description="Type of message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    content: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """Create message from JSON string"""
        return cls.model_validate_json(json_str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return self.model_dump()


class CommunicationProtocol:
    """
    Manages message passing between agents
    Provides logging and validation of agent communications
    """
    
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.message_history: list[AgentMessage] = []
    
    def send_message(
        self,
        sender: str,
        receiver: str,
        message_type: MessageType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Create and log a message between agents
        
        Args:
            sender: Name of sending agent
            receiver: Name of receiving agent
            message_type: Type of message
            content: Message payload
            metadata: Additional context
            
        Returns:
            AgentMessage: The created message
        """
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        if self.enable_logging:
            self.message_history.append(message)
            print(f"[{message.timestamp}] {sender} -> {receiver}: {message_type.value}")
        
        return message
    
    def get_message_history(self) -> list[AgentMessage]:
        """Get all messages sent through this protocol"""
        return self.message_history
    
    def save_history(self, filepath: str):
        """Save message history to file"""
        with open(filepath, 'w') as f:
            history = [msg.to_dict() for msg in self.message_history]
            json.dump(history, f, indent=2)
    
    def clear_history(self):
        """Clear message history"""
        self.message_history.clear()

