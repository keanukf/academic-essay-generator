"""WriterAgent - Generates essay sections following academic conventions."""

from typing import Dict, Any
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.utils.prompts import WRITER_AGENT_SYSTEM_PROMPT, get_writer_prompt


def writer_agent(state: EssayState, ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    WriterAgent generates essay sections one at a time following the outline.
    
    Args:
        state: Current essay state
        ollama_client: Ollama client instance
        
    Returns:
        Updated state dictionary with new sections
    """
    print("✍️  WriterAgent: Generating essay sections...")
    
    if not state.outline or "sections" not in state.outline:
        print("  Warning: No outline available")
        return {"sections": {}}
    
    sections = state.sections.copy() if state.sections else {}
    outline_sections = state.outline.get("sections", [])
    
    try:
        for section_info in outline_sections:
            section_name = section_info.get("name", "Untitled Section")
            
            # Skip if already written (unless we're revising)
            if section_name in sections and state.revision_count == 0:
                print(f"  ⊘ Skipping {section_name} (already written)")
                continue
            
            print(f"  Writing: {section_name}...")
            
            # Generate prompt for this section
            prompt = get_writer_prompt(section_name, section_info, state.research_notes, state.topic)
            
            # Generate section content
            section_content = ollama_client.generate(
                prompt=prompt,
                system=WRITER_AGENT_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=4000
            )
            
            sections[section_name] = section_content
            print(f"    ✓ Completed {section_name} ({len(section_content)} characters)")
        
        print(f"  ✓ Generated {len(sections)} sections total")
        return {"sections": sections}
    
    except Exception as e:
        print(f"  ✗ Error in WriterAgent: {str(e)}")
        return {"sections": sections}

