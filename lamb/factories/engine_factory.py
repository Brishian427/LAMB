"""
Factory for creating decision engines.
"""

from typing import Dict, Any
from ..engines.llm_engine import LLMEngine
from ..engines import MockEngine


class EngineFactory:
    """Factory for creating engines"""
    
    @staticmethod
    def create(execution_mode: str, config: Dict[str, Any]):
        """Create engine based on execution mode"""
        import os
        
        if execution_mode == "llm" and os.getenv("OPENAI_API_KEY"):
            return LLMEngine(
                model=config.get("llm_model", "gpt-3.5-turbo"),
                temperature=config.get("temperature", 0.7)
            )
        else:
            # Use mock engine if no API key or rule mode
            return MockEngine()
