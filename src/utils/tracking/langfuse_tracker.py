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
            self._current_trace = self._client.trace(
                name=name,
                metadata=metadata or {}
            )
            return self._current_trace.id
        except Exception as e:
            print(f"Warning: Failed to start trace: {str(e)}")
            return None
    
    def end_trace(self, trace_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a workflow trace."""
        if not self.is_enabled() or not trace_id:
            return
        
        try:
            # Update metadata if provided
            if metadata and self._current_trace:
                self._current_trace.update(metadata=metadata)
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
            if self._current_trace:
                # Create span under current trace
                span = self._current_trace.span(
                    name=name,
                    metadata=metadata or {}
                )
                span_id = span.id if hasattr(span, 'id') else str(id(span))
                self._active_spans[span_id] = span
                return span_id
            else:
                # No parent trace, create standalone span
                span = self._client.span(
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
                # Remove from active spans (span will be finalized automatically)
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
            
            generation_config = {}
            if temperature is not None:
                generation_config["temperature"] = temperature
            if max_tokens is not None:
                generation_config["max_tokens"] = max_tokens
            
            # Create generation under current trace or standalone
            if self._current_trace:
                generation = self._current_trace.generation(
                    name=name,
                    model=model,
                    input=prompt,
                    output=response,
                    metadata=metadata,
                    **generation_config
                )
            else:
                generation = self._client.generation(
                    name=name,
                    model=model,
                    input=prompt,
                    output=response,
                    metadata=metadata,
                    **generation_config
                )
            
            # Flush to ensure data is sent
            self._client.flush()
        except Exception as e:
            print(f"Warning: Failed to track LLM call: {str(e)}")

