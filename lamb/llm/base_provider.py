"""
Base LLM provider interface for the LAMB framework.

Based on Technical_Specification.md Section 1.3: Engine Decision Flow Algorithm.
Defines the common interface that all LLM providers must implement,
enabling easy switching between different LLM services.

Supported providers:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Future: Anthropic Claude, Google PaLM, Local models, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    text: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    response_time: float = 0.0
    model: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement this interface to ensure
    consistent behavior and easy interchangeability.
    """
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate single response from LLM.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response text
            
        Raises:
            EngineTimeoutError: If request times out or fails
        """
        pass
    
    @abstractmethod
    def generate_batch_response(self, batch_prompt: str, expected_count: int) -> str:
        """
        Generate batch response for multiple agents.
        
        Args:
            batch_prompt: Batch prompt containing multiple agent contexts
            expected_count: Expected number of responses
            
        Returns:
            Generated batch response text
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get provider performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate connection to LLM service.
        
        Returns:
            True if connection is working, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'provider': self.__class__.__name__,
            'model': getattr(self, 'model', 'unknown'),
            'max_tokens': getattr(self, 'max_tokens', 0),
            'temperature': getattr(self, 'temperature', 0.0)
        }
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for given token usage.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD (0.0 if not implemented)
        """
        return 0.0
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model names (empty if not implemented)
        """
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={getattr(self, 'model', 'unknown')})"
