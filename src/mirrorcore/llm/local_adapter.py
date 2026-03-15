"""
Local Adapter

Adapter for local language models (Ollama, etc.).
"""

import subprocess
import json
import time
from typing import Dict, Any, Optional
from .base import LLMBase, LLMResponse


class LocalAdapter(LLMBase):
    """Adapter for local language models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout_seconds", 30)
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate response using local model."""
        enhanced_prompt = self.enhance_prompt(prompt, context)
        
        start_time = time.time()
        
        try:
            # Use Ollama API
            response = self._call_ollama(enhanced_prompt)
            response_time = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=response.get("response", ""),
                confidence=0.8,  # Local models have consistent confidence
                tokens_used=response.get("eval_count", 0),
                response_time_ms=response_time,
                model_info={
                    "model": self.model_name,
                    "provider": "local",
                    "engine": "ollama"
                }
            )
        
        except Exception as e:
            # Fallback response
            return LLMResponse(
                content=f"Local model error: {str(e)}",
                confidence=0.0,
                tokens_used=0,
                response_time_ms=int((time.time() - start_time) * 1000),
                model_info={"error": str(e)}
            )
    
    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API."""
        import urllib.request
        import urllib.error
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        url = f"{self.base_url}/api/generate"
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        
        except urllib.error.URLError as e:
            raise Exception(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        try:
            import urllib.request
            
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                # Check if our model is available
                models = data.get("models", [])
                return any(model.get("name") == self.model_name for model in models)
        
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the local model."""
        return {
            "name": self.model_name,
            "provider": "local",
            "base_url": self.base_url,
            "available": self.is_available(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
