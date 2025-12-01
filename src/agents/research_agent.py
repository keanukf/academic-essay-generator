"""ResearchAgent - Analyzes literature and extracts key information."""

from typing import Dict, Any
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.utils.prompts import RESEARCH_AGENT_SYSTEM_PROMPT, get_research_prompt


def research_agent(state: EssayState, ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    ResearchAgent processes literature chunks and extracts key information.
    
    Args:
        state: Current essay state
        ollama_client: Ollama client instance
        
    Returns:
        Updated state dictionary
    """
    print("🔍 ResearchAgent: Analyzing literature...")
    
    if not state.literature_chunks:
        print("  Warning: No literature chunks available")
        return {"research_notes": {}}
    
    try:
        # Generate prompt
        prompt = get_research_prompt(state.topic, state.literature_chunks)
        
        # Get structured response
        research_notes = ollama_client.generate_structured(
            prompt=prompt,
            system=RESEARCH_AGENT_SYSTEM_PROMPT,
            temperature=0.3
        )
        
        print(f"  ✓ Extracted {len(research_notes.get('arguments', []))} arguments")
        print(f"  ✓ Found {len(research_notes.get('quotes', []))} quotes")
        print(f"  ✓ Identified {len(research_notes.get('themes', []))} themes")
        
        return {"research_notes": research_notes}
    
    except Exception as e:
        print(f"  ✗ Error in ResearchAgent: {str(e)}")
        return {"research_notes": {}}

