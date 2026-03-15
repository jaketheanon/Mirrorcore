"""
Voice Model

Models user's communication style and tone preferences.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VoiceProfile:
    """User voice communication profile."""
    formality: str  # casual, moderate, formal
    directness: str  # indirect, moderate, direct
    technical_detail: str  # minimal, moderate, comprehensive
    tone: str  # encouraging, analytical, instructional, neutral
    confidence: float


class VoiceModel:
    """Models and adapts to user's voice preferences."""
    
    def __init__(self):
        self.current_profile: Optional[VoiceProfile] = None
    
    def load_profile(self, profile_data: Dict[str, Any]) -> VoiceProfile:
        """Load voice profile from data."""
        self.current_profile = VoiceProfile(
            formality=profile_data.get("formality", "moderate"),
            directness=profile_data.get("directness", "moderate"),
            technical_detail=profile_data.get("technical_detail", "moderate"),
            tone=profile_data.get("tone", "neutral"),
            confidence=profile_data.get("confidence", 0.5)
        )
        return self.current_profile
    
    def adapt_response(self, response: str) -> str:
        """Adapt response based on voice profile."""
        if not self.current_profile:
            return response
        
        adapted = response
        
        # Adapt formality
        if self.current_profile.formality == "casual":
            adapted = self._make_casual(adapted)
        elif self.current_profile.formality == "formal":
            adapted = self._make_formal(adapted)
        
        # Adapt detail level
        if self.current_profile.technical_detail == "minimal":
            adapted = self._reduce_detail(adapted)
        elif self.current_profile.technical_detail == "comprehensive":
            adapted = self._enhance_detail(adapted)
        
        return adapted
    
    def _make_casual(self, text: str) -> str:
        """Make text more casual."""
        replacements = {
            "Therefore": "So",
            "However": "But",
            "Furthermore": "Also",
            "consequently": "so",
            ".": "."
        }
        for formal, casual in replacements.items():
            text = text.replace(formal, casual)
        return text
    
    def _make_formal(self, text: str) -> str:
        """Make text more formal."""
        replacements = {
            "So": "Therefore",
            "But": "However",
            "Also": "Furthermore",
            "gonna": "going to",
            "wanna": "want to"
        }
        for casual, formal in replacements.items():
            text = text.replace(casual, formal)
        return text
    
    def _reduce_detail(self, text: str) -> str:
        """Reduce technical detail in text."""
        # Simple implementation - remove detailed explanations
        lines = text.split('\n')
        essential_lines = []
        
        for line in lines:
            if len(line.strip()) < 100:  # Keep shorter lines
                essential_lines.append(line)
        
        return '\n'.join(essential_lines)
    
    def _enhance_detail(self, text: str) -> str:
        """Add more detail to text."""
        # Simple implementation - add explanatory notes
        if "error" in text.lower():
            text += "\n\nNote: This error typically occurs when permissions are misconfigured."
        return text
