"""Service for interacting with LLM models through Ollama and Gemini."""

import logging
import os
from collections.abc import Generator
from typing import Optional

import ollama
from tapio.config.settings import get_gemini_api_key, has_gemini_api_key
from tapio.config.model_config import get_model_config

# Configure logging
logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with LLM models through Ollama and Gemini."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """Initialize the LLM service with the given model settings.

        Args:
            model_name: The name of the model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Get model configuration
        self.model_config = get_model_config(model_name)
        self.provider = self.model_config.provider
        
        # Initialize provider-specific clients
        self._setup_clients()

    def _setup_clients(self) -> None:
        """Setup provider-specific clients."""
        if self.provider == "gemini":
            if not has_gemini_api_key():
                raise ValueError(
                    "Gemini API key required for Gemini models. "
                    "Please set GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_AI_API_KEY environment variable."
                )
            
            try:
                import google.generativeai as genai
                api_key = get_gemini_api_key()
                genai.configure(api_key=api_key)
                self.gemini_client = genai.GenerativeModel(self.model_config.name)
                logger.info(f"Configured Gemini client with model: {self.model_config.name}")
            except ImportError:
                raise ImportError(
                    "google-generativeai package required for Gemini models. "
                    "Install with: uv add google-generativeai"
                )
        elif self.provider == "ollama":
            # Ollama client is initialized as needed
            pass
        
        logger.info(f"Initialized LLM service with model: {self.model_name} (provider: {self.provider})")

    def check_model_availability(self) -> bool:
        """Check if the model is available for the configured provider.

        Returns:
            bool: True if the model is available, False otherwise
        """
        if self.provider == "gemini":
            # For Gemini, check if we have API key and the client is configured
            return has_gemini_api_key() and hasattr(self, 'gemini_client')
            
        elif self.provider == "ollama":
            try:
                models_response = ollama.list()

                if not hasattr(models_response, "models") or not models_response.models:
                    logger.warning("No models found in Ollama")
                    return False

                # Check if the model exists - allow for model name variations like llama3.2:latest
                model_exists = False

                # Extract model names from the Model objects
                available_models = []
                for model_obj in models_response.models:
                    # Each model object has a 'model' attribute with the name
                    model_name = model_obj.model
                    if model_name:  # Ensure we have a valid model name
                        available_models.append(model_name)

                # Log available models
                logger.info(
                    f"Available Ollama models: {', '.join(available_models)}",
                )

                # Check for exact match or base name match (handle :tag variations)
                for model_name in available_models:
                    # Exact match
                    if model_name == self.model_name:
                        model_exists = True
                        logger.info(f"Found exact matching model: {model_name}")
                        break
                    # If user provided base name (no tag), match any variant with tags
                    elif ":" not in self.model_name and model_name.startswith(f"{self.model_name}:"):
                        model_exists = True
                        logger.info(
                            f"Found matching model: {model_name} for base name {self.model_name}",
                        )
                        break
                    # If user provided name with tag, check if base names match
                    elif ":" in self.model_name and ":" in model_name:
                        user_base = self.model_name.split(":")[0]
                        model_base = model_name.split(":")[0]
                        if user_base == model_base:
                            model_exists = True
                            logger.info(
                                f"Found matching model: {model_name} for requested {self.model_name}",
                            )
                            break

                if not model_exists:
                    logger.warning(
                        f"{self.model_name} model not found in Ollama. Please pull it with 'ollama pull {self.model_name}'",
                    )
                    return False
                return True
            except Exception as e:
                logger.warning(f"Could not connect to Ollama: {e}")
                logger.warning("Make sure Ollama is running")
                return False
        
        return False

    def generate_response(self, prompt: str, system_prompt: str | None = None) -> str | dict:
        """Generate a response from the LLM model.

        Args:
            prompt: The prompt to generate a response for
            system_prompt: Optional system prompt to set context

        Returns:
            str: The generated response
        """
        if self.provider == "gemini":
            return self._generate_gemini_response(prompt, system_prompt)
        elif self.provider == "ollama":
            return self._generate_ollama_response(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_gemini_response(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate response using Gemini API."""
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            
            response = self.gemini_client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return f"Error: Could not generate a response from Gemini. Please check your API key and try again."

    def _generate_ollama_response(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate response using Ollama."""
        try:
            messages = []

            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add user message
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating Ollama response: {e}")
            return (
                f"Error: Could not generate a response. "
                f"Please check if Ollama is running with the {self.model_name} model."
            )

    def generate_response_stream(self, prompt: str, system_prompt: str | None = None) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM model.

        Args:
            prompt: The prompt to generate a response for
            system_prompt: Optional system prompt to set context

        Yields:
            str: Chunks of the generated response
        """
        if self.provider == "gemini":
            yield from self._generate_gemini_response_stream(prompt, system_prompt)
        elif self.provider == "ollama":
            yield from self._generate_ollama_response_stream(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_gemini_response_stream(self, prompt: str, system_prompt: str | None = None) -> Generator[str, None, None]:
        """Generate streaming response using Gemini API."""
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            
            response = self.gemini_client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Error generating Gemini streaming response: {e}")
            yield f"Error: Could not generate a streaming response from Gemini. Please check your API key and try again."

    def _generate_ollama_response_stream(self, prompt: str, system_prompt: str | None = None) -> Generator[str, None, None]:
        """Generate streaming response using Ollama."""
        try:
            messages = []

            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add user message
            messages.append({"role": "user", "content": prompt})

            # Use streaming chat with optimized options for faster response
            logger.info("About to call ollama.chat with streaming")
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": 2048,  # Reduce context window for faster processing
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "seed": -1,
                    "num_thread": 0,  # Use all available threads
                },
                stream=True,
                keep_alive="5m",  # Keep model loaded for faster subsequent requests
            )

            logger.info("Starting to iterate over ollama stream")
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    logger.debug(
                        f"LLM yielding chunk of {len(content)} characters",
                    )
                    yield content

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield (
                f"Error: Could not generate a response. "
                f"Please check if Ollama is running with the {self.model_name} model."
            )

    def get_model_name(self) -> str:
        """Get the name of the model being used.

        Returns:
            str: The model name
        """
        return self.model_name
