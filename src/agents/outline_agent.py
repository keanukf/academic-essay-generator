"""OutlineAgent - Creates structured essay outline."""

from typing import Dict, Any
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.utils.prompts import OUTLINE_AGENT_SYSTEM_PROMPT, get_outline_prompt


def outline_agent(state: EssayState, ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    OutlineAgent creates a structured essay outline based on topic and criteria.
    
    Args:
        state: Current essay state
        ollama_client: Ollama client instance
        
    Returns:
        Updated state dictionary
    """
    print("📋 OutlineAgent: Creating essay outline...")
    
    if not state.topic or not state.criteria:
        print("  Warning: Missing topic or criteria")
        return {"outline": {}}
    
    try:
        # Generate prompt
        prompt = get_outline_prompt(state.topic, state.criteria, state.research_notes)
        
        # Get structured response
        outline = ollama_client.generate_structured(
            prompt=prompt,
            system=OUTLINE_AGENT_SYSTEM_PROMPT,
            temperature=0.5
        )
        
        num_sections = len(outline.get("sections", []))
        estimated_words = outline.get("total_estimated_words", 0)
        print(f"  ✓ Created outline with {num_sections} sections")
        print(f"  ✓ Estimated length: {estimated_words} words")
        
        return {"outline": outline}
    
    except Exception as e:
        print(f"  ✗ Error in OutlineAgent: {str(e)}")
        return {"outline": {}}

