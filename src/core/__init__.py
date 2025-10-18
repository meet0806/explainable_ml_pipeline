"""Core modules for the agentic ML pipeline."""

from .base_agent import BaseAgent
from .communication import AgentMessage, MessageType, CommunicationProtocol

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "MessageType",
    "CommunicationProtocol",
]

