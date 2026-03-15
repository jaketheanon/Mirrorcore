"""
LLM Base

Base class for language model interactions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from language model."""
    content: str
    confidence: float
    tokens_used: int
    response_time_ms: int
    model_info: Dict[str, Any]


class LLMBase(ABC):
    """Base class for language model adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model", "default")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate a response from the language model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the language model is available."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    def enhance_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance prompt with context and system instructions."""
        enhanced = f"You are Mirrorcore, a local-first reasoning assistant.\n\n"
        enhanced += f"User request: {prompt}\n\n"
        
        if context:
            enhanced += "Context:\n"
            for key, value in context.items():
                enhanced += f"- {key}: {value}\n"
            enhanced += "\n"
        
        enhanced += "Provide a helpful, accurate response.\n"
        
        return enhanced
    
    def validate_response(self, response: str) -> bool:
        """Validate that response is appropriate."""
        # Basic validation
        if not response or len(response.strip()) == 0:
            return False
        
        # Check for refusal patterns
        refusal_patterns = ["I cannot", "I'm not able", "I don't provide"]
        response_lower = response.lower()
        
        for pattern in refusal_patterns:
            if pattern in response_lower:
                return False
        
        return True
