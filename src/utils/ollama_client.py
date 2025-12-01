"""Ollama client wrapper for LLM inference."""

import json
import time
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path


class OllamaClient:
    """Wrapper for Ollama API calls with error handling and retry logic."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 300):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3.1:8b-instruct-q4_K_M")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
    
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
            result = self._retry_with_backoff(_make_request)
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
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
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}\nResponse: {response_text[:200]}")

