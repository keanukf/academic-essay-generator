"""CitationAgent - Manages references and ensures proper citation format."""

from typing import Dict, Any, List
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.utils.prompts import CITATION_AGENT_SYSTEM_PROMPT, get_citation_prompt


def citation_agent(state: EssayState, ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    CitationAgent processes sections to identify citation needs and format them in APA style.
    
    Args:
        state: Current essay state
        ollama_client: Ollama client instance
        
    Returns:
        Updated state dictionary with citations
    """
    print("📚 CitationAgent: Processing citations...")
    
    if not state.sections:
        print("  Warning: No sections available for citation")
        return {"citations": []}
    
    try:
        # Generate prompt
        prompt = get_citation_prompt(state.sections, state.literature_chunks)
        
        # Get structured response
        citation_data = ollama_client.generate_structured(
            prompt=prompt,
            system=CITATION_AGENT_SYSTEM_PROMPT,
            temperature=0.3
        )
        
        citations = citation_data.get("citations", [])
        bibliography = citation_data.get("bibliography", [])
        
        # Combine citations and bibliography
        all_citations = citations + [{"type": "bibliography", **bib} for bib in bibliography]
        
        print(f"  ✓ Identified {len(citations)} citation points")
        print(f"  ✓ Created bibliography with {len(bibliography)} entries")
        
        return {"citations": all_citations}
    
    except Exception as e:
        print(f"  ✗ Error in CitationAgent: {str(e)}")
        return {"citations": []}

