"""Abstract base class for tracking implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseTracker(ABC):
    """Abstract interface for tracking agent steps and LLM calls."""
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        pass
    
    @abstractmethod
    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Start a workflow trace.
        
        Args:
            name: Trace name
            metadata: Optional metadata dictionary
            
        Returns:
            Trace ID (or None if tracking disabled)
        """
        pass
    
    @abstractmethod
    def end_trace(self, trace_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End a workflow trace.
        
        Args:
            trace_id: Trace ID from start_trace
            metadata: Optional metadata dictionary
        """
        pass
    
    @abstractmethod
    def start_span(self, name: str, parent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Start an agent span.
        
        Args:
            name: Span name (e.g., agent name)
            parent_id: Parent trace/span ID
            metadata: Optional metadata dictionary
            
        Returns:
            Span ID (or None if tracking disabled)
        """
        pass
    
    @abstractmethod
    def end_span(self, span_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End an agent span.
        
        Args:
            span_id: Span ID from start_span
            metadata: Optional metadata dictionary
        """
        pass
    
    @abstractmethod
    def track_llm_call(
        self,
        name: str,
        prompt: str,
        response: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track an LLM API call.
        
        Args:
            name: Call name/identifier
            prompt: Input prompt
            response: Generated response
            parent_id: Parent trace/span ID
            metadata: Optional metadata (model, temperature, tokens, etc.)
        """
        pass

