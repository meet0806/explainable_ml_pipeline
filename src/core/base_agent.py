"""
Base Agent Class
All specialized agents inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from datetime import datetime

from .communication import AgentMessage, MessageType, CommunicationProtocol


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the pipeline
    Defines common interface and functionality
    """
    
    def __init__(
        self,
        agent_name: str,
        config: Dict[str, Any],
        communication_protocol: CommunicationProtocol
    ):
        """
        Initialize base agent
        
        Args:
            agent_name: Unique identifier for this agent
            config: Configuration dictionary
            communication_protocol: Shared communication protocol
        """
        self.agent_name = agent_name
        self.config = config
        self.communication = communication_protocol
        self.state: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        
        # Setup logging
        self.logger = logging.getLogger(agent_name)
        self.logger.setLevel(logging.INFO)
        
        # LLM reasoning placeholder
        self.llm_enabled = config.get("llm", {}).get("reasoning_enabled", False)
        
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - must be implemented by each agent
        
        Args:
            input_data: Input data from previous agent or orchestrator
            
        Returns:
            Dict containing agent results
        """
        pass
    
    def receive_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process incoming message and execute agent logic
        
        Args:
            message: Incoming message from another agent
            
        Returns:
            Response message
        """
        self.logger.info(f"Received message from {message.sender}")
        
        try:
            # Execute agent logic
            results = self.execute(message.content)
            
            # Send success response
            response = self.communication.send_message(
                sender=self.agent_name,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                content=results,
                metadata={"status": "success", "timestamp": datetime.now().isoformat()}
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error executing agent: {str(e)}")
            
            # Send error response
            error_response = self.communication.send_message(
                sender=self.agent_name,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                content={"error": str(e)},
                metadata={"status": "error"}
            )
            
            return error_response
    
    def send_message(
        self,
        receiver: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST
    ) -> AgentMessage:
        """
        Send message to another agent
        
        Args:
            receiver: Target agent name
            content: Message content
            message_type: Type of message
            
        Returns:
            Created message
        """
        return self.communication.send_message(
            sender=self.agent_name,
            receiver=receiver,
            message_type=message_type,
            content=content
        )
    
    def llm_reason(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        LLM-based reasoning using Ollama
        
        Args:
            prompt: Question or task for the LLM
            context: Additional context data
            
        Returns:
            LLM response string
        """
        if not self.llm_enabled:
            return "LLM reasoning disabled"
        
        try:
            from ollama import Client
            
            # Initialize Ollama client
            client = Client()
            
            # Format prompt with context
            full_prompt = f"{prompt}\n\nContext:\n{str(context)[:500]}"  # Limit context size
            
            # Get model from config
            model = self.config.get("llm", {}).get("model", "llama3.1:8b")
            
            # Call Ollama
            response = client.generate(
                model=model,
                prompt=full_prompt,
                options={
                    "temperature": self.config.get("llm", {}).get("temperature", 0.7),
                }
            )
            
            self.logger.info(f"LLM reasoning completed with {model}")
            return response['response']
            
        except Exception as e:
            self.logger.error(f"LLM reasoning failed: {e}")
            return f"LLM error: {str(e)}"
    
    def save_state(self):
        """Save agent state for reproducibility"""
        self.state["last_update"] = datetime.now().isoformat()
        self.state["results"] = self.results
        
    def get_results(self) -> Dict[str, Any]:
        """Get agent execution results"""
        return self.results
    
    def reset(self):
        """Reset agent state"""
        self.state = {}
        self.results = {}
        self.logger.info(f"{self.agent_name} reset")

