"""LangGraph workflow definition for essay generation pipeline."""

import time
from typing import Literal, Optional, TYPE_CHECKING, Callable, Dict, Any
from langgraph.graph import StateGraph, END
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.agents.research_agent import research_agent
from src.agents.outline_agent import outline_agent
from src.agents.writer_agent import writer_agent
from src.agents.citation_agent import citation_agent
from src.agents.review_agent import review_agent
from src.agents.editor_agent import editor_agent

if TYPE_CHECKING:
    from src.utils.tracking.base_tracker import BaseTracker


def create_workflow(
    ollama_client: OllamaClient,
    review_threshold: float = 0.7,
    max_revision_cycles: int = 2,
    tracker: Optional["BaseTracker"] = None
):
    """
    Create LangGraph workflow for essay generation.
    
    Args:
        ollama_client: Ollama client instance
        review_threshold: Minimum review score to pass (0.0-1.0)
        max_revision_cycles: Maximum number of revision loops
        tracker: Optional tracker instance for observability
        
    Returns:
        Compiled LangGraph workflow
    """
    # Create state graph
    workflow = StateGraph(EssayState)
    
    def _wrap_agent(
        agent_name: str,
        agent_func: Callable[[EssayState, OllamaClient], Dict[str, Any]]
    ) -> Callable[[EssayState], EssayState]:
        """Wrap an agent function with tracking using context managers."""
        def tracked_node(state: EssayState) -> EssayState:
            start_time = time.time()
            
            # Create input state summary
            input_metadata = {
                "topic": state.topic[:100] if state.topic else "",
                "revision_count": state.revision_count,
                "review_score": state.review_score,
                "has_outline": bool(state.outline),
                "has_sections": bool(state.sections),
                "sections_count": len(state.sections) if state.sections else 0,
                "citations_count": len(state.citations) if state.citations else 0
            }
            
            # Use context manager if available (simplified approach)
            if tracker and tracker.is_enabled() and hasattr(tracker, 'span_context'):
                try:
                    with tracker.span_context(name=agent_name, metadata=input_metadata) as span:
                        # Execute agent
                        updates = agent_func(state, ollama_client)
                        result_state = state.model_copy(update=updates)
                        
                        # Update span with output metadata
                        if span:
                            execution_time = time.time() - start_time
                            output_metadata = {
                                "execution_time_seconds": execution_time,
                                "success": True
                            }
                            # Add output summary based on agent
                            if agent_name == "research":
                                research_notes = updates.get("research_notes", {})
                                output_metadata["arguments_count"] = len(research_notes.get("arguments", []))
                                output_metadata["quotes_count"] = len(research_notes.get("quotes", []))
                                output_metadata["themes_count"] = len(research_notes.get("themes", []))
                            elif agent_name == "outline":
                                outline = updates.get("outline", {})
                                output_metadata["sections_count"] = len(outline.get("sections", []))
                            elif agent_name == "writer":
                                sections = updates.get("sections", {})
                                output_metadata["sections_generated"] = len(sections)
                            elif agent_name == "citation":
                                citations = updates.get("citations", [])
                                output_metadata["citations_added"] = len(citations)
                            elif agent_name == "review":
                                output_metadata["review_score"] = updates.get("review_score", 0.0)
                                output_metadata["feedback_count"] = len(updates.get("review_feedback", []))
                            
                            span.update(metadata=output_metadata)
                        
                        return result_state
                except Exception as e:
                    # Error is logged by span context manager
                    raise
            else:
                # Fallback: execute without tracking
                updates = agent_func(state, ollama_client)
                return state.model_copy(update=updates)
        
        return tracked_node
    
    # Define node functions that wrap agents with tracking
    def research_node(state: EssayState) -> EssayState:
        return _wrap_agent("research", research_agent)(state)
    
    def outline_node(state: EssayState) -> EssayState:
        return _wrap_agent("outline", outline_agent)(state)
    
    def writer_node(state: EssayState) -> EssayState:
        return _wrap_agent("writer", writer_agent)(state)
    
    def citation_node(state: EssayState) -> EssayState:
        return _wrap_agent("citation", citation_agent)(state)
    
    def review_node(state: EssayState) -> EssayState:
        return _wrap_agent("review", review_agent)(state)
    
    def editor_node(state: EssayState) -> EssayState:
        return _wrap_agent("editor", editor_agent)(state)
    
    def increment_revision_node(state: EssayState) -> EssayState:
        """Increment revision count when routing back to writer."""
        print(f"🔄 Incrementing revision count: {state.revision_count + 1}")
        return state.model_copy(update={"revision_count": state.revision_count + 1})
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("outline", outline_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("citation", citation_node)
    workflow.add_node("review", review_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("increment_revision", increment_revision_node)
    
    # Define conditional edge function for review loop
    def should_revise(state: EssayState) -> Literal["revise", "finalize"]:
        """Determine if essay needs revision or can be finalized."""
        revision_count = state.revision_count
        review_score = state.review_score
        
        # If score is below threshold and we haven't exceeded max revisions
        if review_score < review_threshold and revision_count < max_revision_cycles:
            return "revise"
        else:
            return "finalize"
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add linear edges
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "writer")
    workflow.add_edge("writer", "citation")
    workflow.add_edge("citation", "review")
    
    # Conditional edge after review
    workflow.add_conditional_edges(
        "review",
        should_revise,
        {
            "revise": "increment_revision",  # Increment count first
            "finalize": "editor"  # Proceed to editor
        }
    )
    
    # Route from increment_revision to writer
    workflow.add_edge("increment_revision", "writer")
    
    # Final edge
    workflow.add_edge("editor", END)
    
    # Compile workflow
    return workflow.compile()

