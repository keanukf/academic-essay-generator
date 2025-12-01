"""Pydantic state model for essay generation workflow."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class EssayState(BaseModel):
    """Shared state for the essay generation pipeline."""
    
    topic: str = Field(default="", description="Essay topic")
    criteria: str = Field(default="", description="Evaluation criteria")
    literature_chunks: List[str] = Field(default_factory=list, description="Chunked text from literature PDFs")
    research_notes: Dict[str, Any] = Field(default_factory=dict, description="Structured research notes with arguments, quotes, themes")
    outline: Dict[str, Any] = Field(default_factory=dict, description="Structured essay outline")
    sections: Dict[str, str] = Field(default_factory=dict, description="Generated essay sections (section_name -> content)")
    citations: List[Dict[str, str]] = Field(default_factory=list, description="APA-formatted citations")
    review_feedback: List[str] = Field(default_factory=list, description="Review feedback and suggestions")
    final_essay: str = Field(default="", description="Final polished essay in Markdown format")
    revision_count: int = Field(default=0, description="Number of revision cycles completed")
    review_score: float = Field(default=0.0, description="Review score (0.0-1.0)")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

