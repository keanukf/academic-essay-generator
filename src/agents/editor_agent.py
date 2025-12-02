"""EditorAgent - Final polish for coherence and academic tone."""

from typing import Dict, Any
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.utils.prompts import EDITOR_AGENT_SYSTEM_PROMPT, get_editor_prompt


def editor_agent(state: EssayState, ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    EditorAgent performs final editing and combines sections into polished essay.
    
    Args:
        state: Current essay state
        ollama_client: Ollama client instance
        
    Returns:
        Updated state dictionary with final essay
    """
    print("✨ EditorAgent: Final editing and formatting...")
    
    if not state.sections:
        print("  Warning: No sections available for editing")
        return {"final_essay": ""}
    
    try:
        # Calculate input word count from all sections
        input_word_count = sum(len(content.split()) for content in state.sections.values())
        print(f"  📊 Input sections: {input_word_count} words across {len(state.sections)} sections")
        
        # Generate prompt
        prompt = get_editor_prompt(state.sections, state.citations, state.review_feedback)
        
        # Generate final essay
        final_essay = ollama_client.generate(
            prompt=prompt,
            system=EDITOR_AGENT_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=8000
        )
        
        # Add bibliography if citations exist
        if state.citations:
            bibliography = [c for c in state.citations if c.get("type") == "bibliography"]
            if bibliography:
                final_essay += "\n\n## References\n\n"
                for bib in bibliography:
                    author = bib.get("author", "Unknown")
                    year = bib.get("year", "n.d.")
                    title = bib.get("title", "Untitled")
                    publisher = bib.get("publisher", "")
                    final_essay += f"{author} ({year}). {title}. {publisher}\n\n"
        
        # Calculate output word count
        output_word_count = len(final_essay.split())
        
        # Validate content preservation
        compression_ratio = output_word_count / input_word_count if input_word_count > 0 else 0
        
        print(f"  📊 Output essay: {output_word_count} words")
        print(f"  📊 Content preservation: {compression_ratio:.1%} ({output_word_count}/{input_word_count} words)")
        
        if compression_ratio < 0.7:
            print(f"  ⚠️  WARNING: Significant content compression detected! Output is only {compression_ratio:.1%} of input.")
            print(f"  ⚠️  This suggests the model may have summarized/compressed content instead of preserving it.")
        elif compression_ratio < 0.9:
            print(f"  ⚠️  WARNING: Moderate content reduction detected ({compression_ratio:.1%}).")
        else:
            print(f"  ✓ Content preservation: Good ({compression_ratio:.1%})")
        
        print(f"  ✓ Formatted as Markdown")
        
        return {"final_essay": final_essay}
    
    except Exception as e:
        print(f"  ✗ Error in EditorAgent: {str(e)}")
        return {"final_essay": ""}

