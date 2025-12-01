"""Utility functions for saving intermediate essay checkpoints."""

from pathlib import Path
from typing import Dict, Any, Optional
from src.state.state import EssayState
import json
from datetime import datetime


def save_checkpoint(state: EssayState, checkpoint_dir: Path, step_name: str) -> Path:
    """
    Save an intermediate checkpoint of the essay state.
    
    Args:
        state: Current essay state
        checkpoint_dir: Directory to save checkpoints
        step_name: Name of the step (e.g., "outline", "writer", "citation")
        
    Returns:
        Path to the saved checkpoint file
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint data
    checkpoint_data = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "revision_count": state.revision_count,
        "review_score": state.review_score,
        "state": state.model_dump()
    }
    
    # Save as JSON
    checkpoint_file = checkpoint_dir / f"checkpoint_{step_name}_rev{state.revision_count}.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    return checkpoint_file


def save_intermediate_essay(state: EssayState, checkpoint_dir: Path, step_name: str) -> Optional[Path]:
    """
    Save an intermediate version of the essay (if sections are available).
    
    Args:
        state: Current essay state
        checkpoint_dir: Directory to save checkpoints
        step_name: Name of the step
        
    Returns:
        Path to the saved essay file, or None if no sections available
    """
    if not state.sections:
        return None
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Build essay from sections
    essay_parts = []
    
    # Add title if available
    if state.topic:
        essay_parts.append(f"# {state.topic}\n\n")
    
    # Add sections in order
    if state.outline and "sections" in state.outline:
        for section_info in state.outline.get("sections", []):
            section_name = section_info.get("name", "")
            if section_name in state.sections:
                essay_parts.append(f"## {section_name}\n\n")
                essay_parts.append(state.sections[section_name])
                essay_parts.append("\n\n")
    
    # Add citations if available
    if state.citations:
        essay_parts.append("\n\n## References\n\n")
        for citation in state.citations:
            if isinstance(citation, dict):
                author = citation.get("author", "Unknown")
                year = citation.get("year", "n.d.")
                title = citation.get("title", "")
                essay_parts.append(f"- {author} ({year}). {title}\n")
    
    essay_content = "".join(essay_parts)
    
    # Save intermediate essay
    essay_file = checkpoint_dir / f"essay_{step_name}_rev{state.revision_count}.md"
    with open(essay_file, "w", encoding="utf-8") as f:
        f.write(essay_content)
    
    return essay_file


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """
    Load a checkpoint from a JSON file.
    
    Args:
        checkpoint_file: Path to checkpoint JSON file
        
    Returns:
        Checkpoint data dictionary
    """
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return json.load(f)

