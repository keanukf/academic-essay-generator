"""OutlineAgent - Creates structured essay outline."""

from typing import Dict, Any
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.utils.prompts import (
    OUTLINE_AGENT_SYSTEM_PROMPT,
    get_initial_outline_prompt,
    get_outline_refinement_prompt
)


def outline_agent(state: EssayState, ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    OutlineAgent creates a structured essay outline in two steps:
    1. First, generates an independent outline based on topic and criteria
    2. Then, refines it by incorporating research findings
    
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
    
    target_length = state.target_length
    
    try:
        # Step 1: Generate initial outline independently (without research findings)
        print("  Step 1: Generating initial outline structure...")
        initial_prompt = get_initial_outline_prompt(
            state.topic,
            state.criteria,
            target_length=target_length
        )
        
        initial_outline = ollama_client.generate_structured(
            prompt=initial_prompt,
            system=OUTLINE_AGENT_SYSTEM_PROMPT,
            temperature=0.5
        )
        
        num_sections = len(initial_outline.get("sections", []))
        estimated_words = initial_outline.get("total_estimated_words", 0)
        print(f"  ✓ Initial outline: {num_sections} sections, ~{estimated_words} words")
        
        # Step 2: Refine outline by incorporating research findings
        if state.research_notes and any(state.research_notes.values()):
            print("  Step 2: Refining outline with research findings...")
            refinement_prompt = get_outline_refinement_prompt(
                state.topic,
                state.criteria,
                initial_outline,
                state.research_notes,
                target_length=target_length
            )
            
            refined_outline = ollama_client.generate_structured(
                prompt=refinement_prompt,
                system=OUTLINE_AGENT_SYSTEM_PROMPT,
                temperature=0.5
            )
            
            # Use refined outline if it's valid, otherwise fall back to initial
            if refined_outline and refined_outline.get("sections"):
                outline = refined_outline
                print(f"  ✓ Refined outline: {len(outline.get('sections', []))} sections")
            else:
                outline = initial_outline
                print("  ⚠ Using initial outline (refinement produced invalid result)")
        else:
            outline = initial_outline
            print("  ⚠ No research notes available, using initial outline only")
        
        final_estimated_words = outline.get("total_estimated_words", 0)
        print(f"  ✓ Final outline: {len(outline.get('sections', []))} sections, ~{final_estimated_words} words")
        
        return {"outline": outline}
    
    except Exception as e:
        print(f"  ✗ Error in OutlineAgent: {str(e)}")
        return {"outline": {}}

