"""Langfuse implementation of BaseTracker for cloud-hosted tracking."""

import os
from typing import Dict, Any, Optional
from contextlib import contextmanager
from langfuse import Langfuse
from src.utils.tracking.base_tracker import BaseTracker


class LangfuseTracker(BaseTracker):
    """Simplified Langfuse cloud-hosted tracking implementation using context managers."""
    
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
            self._current_trace_context = None
            return
        
        # Get credentials from args or environment
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        
        if not public_key or not secret_key:
            # If credentials not provided, disable tracking
            self._enabled = False
            self._client = None
            self._current_trace_context = None
            return
        
        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            self._current_trace_context = None
        except Exception as e:
            # If initialization fails, disable tracking
            print(f"Warning: Failed to initialize Langfuse tracker: {str(e)}")
            self._enabled = False
            self._client = None
            self._current_trace_context = None
    
    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._enabled and self._client is not None
    
    @contextmanager
    def trace_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for a trace (root span)."""
        if not self.is_enabled():
            yield None
            return
        
        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name=name,
                metadata=metadata or {}
            ) as trace:
                self._current_trace_context = trace
                try:
                    yield trace
                finally:
                    self._current_trace_context = None
                    self._client.flush()
        except Exception as e:
            print(f"Warning: Failed to create trace: {str(e)}")
            yield None
    
    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start a workflow trace. Returns a context manager."""
        # This method is kept for compatibility but should use trace_context() instead
        if not self.is_enabled():
            return None
        # Return a placeholder - actual trace management should use trace_context()
        return "trace_active"
    
    def end_trace(self, trace_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a workflow trace."""
        # This is handled automatically by context manager
        if self.is_enabled():
            self._client.flush()
    
    @contextmanager
    def span_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for an agent span."""
        if not self.is_enabled():
            yield None
            return
        
        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name=name,
                metadata=metadata or {}
            ) as span:
                yield span
        except Exception as e:
            print(f"Warning: Failed to create span: {str(e)}")
            yield None
    
    def start_span(self, name: str, parent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start an agent span. Returns span ID for compatibility."""
        if not self.is_enabled():
            return None
        # Return placeholder - actual span management should use span_context()
        return f"span_{name}"
    
    def end_span(self, span_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """End an agent span."""
        # This is handled automatically by context manager
        if self.is_enabled():
            self._client.flush()
    
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
            # Get model from metadata (don't pop to avoid modifying original)
            model = metadata.get("model", "unknown")
            
            # All metadata (including temperature, max_tokens) goes in the metadata dict
            # Langfuse will handle temperature/max_tokens from metadata
            generation_metadata = metadata.copy()
            
            # Create generation observation - all config goes in metadata
            with self._client.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=prompt,
                output=response,
                metadata=generation_metadata
            ):
                pass  # Context manager handles the observation lifecycle
            
            # Flush to ensure data is sent
            self._client.flush()
        except Exception as e:
            print(f"Warning: Failed to track LLM call: {str(e)}")

