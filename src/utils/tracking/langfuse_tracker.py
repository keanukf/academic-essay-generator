"""Langfuse implementation of BaseTracker for cloud-hosted tracking."""

import os
from typing import Dict, Any, Optional
from langfuse import Langfuse
from src.utils.tracking.base_tracker import BaseTracker


class LangfuseTracker(BaseTracker):
    """Langfuse cloud-hosted tracking implementation."""
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True
    ):
        """
        Initialize Langfuse tracker.
        
        Args:
            public_key: Langfuse public key (or from LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or from LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL
            enabled: Whether tracking is enabled
        """
        self._enabled = enabled
        
        if not enabled:
            self._client = None
            self._current_trace = None
            return
        
        # Get credentials from args or environment
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        
        if not public_key or not secret_key:
            # If credentials not provided, disable tracking
            self._enabled = False
            self._client = None
            self._current_trace = None
            return
        
        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            self._current_trace = None
            self._active_spans = {}  # Store active spans by ID
        except Exception as e:
            # If initialization fails, disable tracking
            print(f"Warning: Failed to initialize Langfuse tracker: {str(e)}")
            self._enabled = False
            self._client = None
            self._current_trace = None
            self._active_spans = {}
    
    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._enabled and self._client is not None
    
    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start a workflow trace."""
        if not self.is_enabled():
            return None
        
        try:
            # In Langfuse, traces are created implicitly by the first span
            # Create a root span which will serve as the trace
            root_span = self._client.start_span(
                name=name,
                metadata=metadata or {}
            )
            # Store the root span as the current trace
            self._current_trace = root_span
            trace_id = root_span.id if hasattr(root_span, 'id') else str(id(root_span))
            return trace_id
        except Exception as e:
            print(f"Warning: Failed to start trace: {str(e)}")
            return None
    
    def end_trace(self, trace_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a workflow trace."""
        if not self.is_enabled() or not trace_id:
            return
        
        try:
            # Update metadata if provided and end the root span (which represents the trace)
            if self._current_trace:
                if metadata:
                    self._current_trace.update(metadata=metadata)
                # End the root span
                self._current_trace.end()
            # Flush to ensure all data is sent
            self._client.flush()
            # Clear current trace
            self._current_trace = None
            self._active_spans.clear()
        except Exception as e:
            print(f"Warning: Failed to end trace: {str(e)}")
    
    def start_span(self, name: str, parent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start an agent span."""
        if not self.is_enabled():
            return None
        
        try:
            # If we have a current trace, create a child span
            if self._current_trace:
                # Get trace_id from the root span
                trace_id = self._current_trace.id if hasattr(self._current_trace, 'id') else None
                span = self._client.start_span(
                    name=name,
                    trace_context={"trace_id": trace_id} if trace_id else None,
                    metadata=metadata or {}
                )
            else:
                # No parent trace, create standalone span (which creates a new trace)
                span = self._client.start_span(
                    name=name,
                    metadata=metadata or {}
                )
            
            span_id = span.id if hasattr(span, 'id') else str(id(span))
            self._active_spans[span_id] = span
            return span_id
        except Exception as e:
            print(f"Warning: Failed to start span: {str(e)}")
            return None
    
    def end_span(self, span_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """End an agent span."""
        if not self.is_enabled() or not span_id:
            return
        
        try:
            span = self._active_spans.get(span_id)
            if span:
                # Update metadata if provided
                if metadata:
                    span.update(metadata=metadata)
                # End the span
                span.end()
                # Remove from active spans
                del self._active_spans[span_id]
                # Flush to ensure data is sent
                self._client.flush()
        except Exception as e:
            print(f"Warning: Failed to end span: {str(e)}")
            # Clean up span from active dict even if update failed
            if span_id in self._active_spans:
                del self._active_spans[span_id]
    
    def track_llm_call(
        self,
        name: str,
        prompt: str,
        response: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track an LLM API call."""
        if not self.is_enabled():
            return
        
        try:
            metadata = metadata or {}
            model = metadata.get("model", "unknown")
            temperature = metadata.get("temperature")
            max_tokens = metadata.get("max_tokens")
            
            # Get trace_id if we have a current trace
            trace_id = None
            if self._current_trace:
                trace_id = self._current_trace.id if hasattr(self._current_trace, 'id') else None
            
            # Create generation using start_as_current_observation
            generation = self._client.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=prompt,
                output=response,
                metadata=metadata,
                trace_context={"trace_id": trace_id} if trace_id else None,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # End the generation observation
            generation.end()
            
            # Flush to ensure data is sent
            self._client.flush()
        except Exception as e:
            print(f"Warning: Failed to track LLM call: {str(e)}")

