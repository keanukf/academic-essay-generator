"""Ollama client wrapper for LLM inference."""

import json
import time
import requests
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from src.utils.tracking.base_tracker import BaseTracker


class OllamaClient:
    """Wrapper for Ollama API calls with error handling and retry logic."""
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        tracker: Optional["BaseTracker"] = None
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3.1:8b-instruct-q4_K_M")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            tracker: Optional tracker instance for observability
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.tracker = tracker
    
    def _check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _retry_with_backoff(self, func, max_retries: int = 3, initial_delay: float = 1.0):
        """Retry function with exponential backoff."""
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        
        if last_exception:
            raise last_exception
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            ConnectionError: If Ollama is not accessible
            requests.RequestException: For API errors
        """
        if not self._check_connection():
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running and the model is available."
            )
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        def _make_request():
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        try:
            start_time = time.time()
            result = self._retry_with_backoff(_make_request)
            response_text = result.get("message", {}).get("content", "")
            latency = time.time() - start_time
            
            # Track LLM call if tracker is available
            if self.tracker and self.tracker.is_enabled():
                # Combine system and user prompts for tracking
                full_prompt = prompt
                if system:
                    full_prompt = f"System: {system}\n\nUser: {prompt}"
                
                metadata = {
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "latency_seconds": latency,
                    "base_url": self.base_url
                }
                
                # Extract token usage if available
                if "prompt_eval_count" in result:
                    metadata["prompt_tokens"] = result.get("prompt_eval_count")
                if "eval_count" in result:
                    metadata["completion_tokens"] = result.get("eval_count")
                if "total_duration" in result:
                    metadata["total_duration_ns"] = result.get("total_duration")
                
                self.tracker.track_llm_call(
                    name="ollama_generate",
                    prompt=full_prompt,
                    response=response_text,
                    metadata=metadata
                )
            
            return response_text
        except requests.exceptions.RequestException as e:
            # Track error if tracker is available
            if self.tracker and self.tracker.is_enabled():
                full_prompt = prompt
                if system:
                    full_prompt = f"System: {system}\n\nUser: {prompt}"
                
                self.tracker.track_llm_call(
                    name="ollama_generate",
                    prompt=full_prompt,
                    response=f"ERROR: {str(e)}",
                    metadata={
                        "model": self.model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "error": str(e),
                        "base_url": self.base_url
                    }
                )
            raise ConnectionError(f"Ollama API error: {str(e)}")
    
    def generate_structured(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using Ollama API.
        
        Args:
            prompt: User prompt with JSON format instructions
            system: System prompt (optional)
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ConnectionError: If Ollama is not accessible
            ValueError: If JSON parsing fails
        """
        # Add JSON format instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no additional text."
        
        response_text = self.generate(
            prompt=json_prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed_json = json.loads(response_text)
            
            # Track structured generation if tracker is available
            if self.tracker and self.tracker.is_enabled():
                full_prompt = prompt
                if system:
                    full_prompt = f"System: {system}\n\nUser: {prompt}"
                
                metadata = {
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "type": "structured_json",
                    "base_url": self.base_url
                }
                
                self.tracker.track_llm_call(
                    name="ollama_generate_structured",
                    prompt=full_prompt,
                    response=json.dumps(parsed_json, indent=2),
                    metadata=metadata
                )
            
            return parsed_json
        except json.JSONDecodeError as e:
            # Track parsing error if tracker is available
            if self.tracker and self.tracker.is_enabled():
                full_prompt = prompt
                if system:
                    full_prompt = f"System: {system}\n\nUser: {prompt}"
                
                self.tracker.track_llm_call(
                    name="ollama_generate_structured",
                    prompt=full_prompt,
                    response=f"PARSE_ERROR: {str(e)}\nResponse: {response_text[:200]}",
                    metadata={
                        "model": self.model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "type": "structured_json",
                        "error": str(e),
                        "base_url": self.base_url
                    }
                )
            raise ValueError(f"Failed to parse JSON response: {str(e)}\nResponse: {response_text[:200]}")

