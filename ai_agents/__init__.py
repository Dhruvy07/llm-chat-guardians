"""
AI Agents Package
Three intelligent AI agents for chatbot enhancement:
1. Security Agent - Malicious content detection
2. Context Agent - Query relevance analysis  
3. Model Selection Agent - Optimal LLM selection
"""

from .security_agent import SecurityAgent
from .context_agent import ContextAgent
from .model_selection_agent import ModelSelectionAgent

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    "SecurityAgent",
    "ContextAgent", 
    "ModelSelectionAgent"
]
