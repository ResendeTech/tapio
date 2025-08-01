"""Global configuration settings for the migri-assistant application.

This module contains common configuration settings used across different
components of the migri-assistant application, including default directories
for storing crawled and parsed content.
"""

import os
from pathlib import Path


# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ.setdefault(key, value)
        except Exception:
            pass  # Silently ignore errors loading .env file

# Load .env on import
load_env_file()

DEFAULT_CONTENT_DIR = "content"

# Default directory paths
DEFAULT_DIRS = {
    "CRAWLED_DIR": "crawled",
    "PARSED_DIR": "parsed",
    "CHROMA_DIR": "chroma_db",
}

DEFAULT_CHROMA_COLLECTION = "tapio_knowledge"
DEFAULT_CRAWLER_TIMEOUT = 30

# API Keys and Authentication
def get_gemini_api_key() -> str | None:
    """Get Gemini API key from environment variables.
    
    Checks the following environment variables in order:
    1. GEMINI_API_KEY
    2. GOOGLE_API_KEY
    3. GOOGLE_AI_API_KEY
    
    Returns:
        The API key if found, None otherwise.
    """
    return (
        os.getenv("GEMINI_API_KEY") or 
        os.getenv("GOOGLE_API_KEY") or 
        os.getenv("GOOGLE_AI_API_KEY")
    )

def has_gemini_api_key() -> bool:
    """Check if a Gemini API key is available."""
    return get_gemini_api_key() is not None

# Model Configuration
DEFAULT_MODELS = {
    "local": "llama3.2",      # Default local model via Ollama
    "cloud": "gemini-2.0-flash",  # Default cloud model via Gemini API
}

def get_default_model() -> str:
    """Get the default model based on available API keys.
    
    Returns:
        Cloud model if Gemini API key is available, otherwise local model.
    """
    # Check if user has explicitly set a default model
    user_default = os.getenv("TAPIO_DEFAULT_MODEL")
    if user_default:
        return user_default
    
    # Auto-select based on available API keys
    if has_gemini_api_key():
        return DEFAULT_MODELS["cloud"]
    return DEFAULT_MODELS["local"]
