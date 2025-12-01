"""LangGraph workflow definition for essay generation pipeline."""

from typing import Literal
from langgraph.graph import StateGraph, END
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.agents.research_agent import research_agent
from src.agents.outline_agent import outline_agent
from src.agents.writer_agent import writer_agent
from src.agents.citation_agent import citation_agent
from src.agents.review_agent import review_agent
from src.agents.editor_agent import editor_agent


def create_workflow(ollama_client: OllamaClient, review_threshold: float = 0.7, max_revision_cycles: int = 2):
    """
    Create LangGraph workflow for essay generation.
    
    Args:
        ollama_client: Ollama client instance
        review_threshold: Minimum review score to pass (0.0-1.0)
        max_revision_cycles: Maximum number of revision loops
        
    Returns:
        Compiled LangGraph workflow
    """
    # Create state graph
    workflow = StateGraph(EssayState)
    
    # Define node functions that wrap agents
    def research_node(state: EssayState) -> EssayState:
        updates = research_agent(state, ollama_client)
        return state.model_copy(update=updates)
    
    def outline_node(state: EssayState) -> EssayState:
        updates = outline_agent(state, ollama_client)
        return state.model_copy(update=updates)
    
    def writer_node(state: EssayState) -> EssayState:
        updates = writer_agent(state, ollama_client)
        return state.model_copy(update=updates)
    
    def citation_node(state: EssayState) -> EssayState:
        updates = citation_agent(state, ollama_client)
        return state.model_copy(update=updates)
    
    def review_node(state: EssayState) -> EssayState:
        updates = review_agent(state, ollama_client)
        return state.model_copy(update=updates)
    
    def editor_node(state: EssayState) -> EssayState:
        updates = editor_agent(state, ollama_client)
        return state.model_copy(update=updates)
    
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

