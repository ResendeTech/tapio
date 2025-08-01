"""Model configuration for ADK agents."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for different LLM models."""
    name: str
    provider: str  # "ollama", "gemini", "openai"
    max_tokens: int = 1024
    temperature: float = 0.7
    context_window: int = 4096
    
    
# Predefined model configurations
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Ollama models (local)
    "llama3.2": ModelConfig(
        name="llama3.2:latest",
        provider="ollama",
        max_tokens=1024,
        temperature=0.7,
        context_window=4096,
    ),
    "llama3.1": ModelConfig(
        name="llama3.1:latest",
        provider="ollama",
        max_tokens=1024,
        temperature=0.7,
        context_window=8192,
    ),
    "mistral": ModelConfig(
        name="mistral:latest",
        provider="ollama",
        max_tokens=1024,
        temperature=0.7,
        context_window=4096,
    ),
    
    # Gemini models (Google Cloud)
    "gemini-2.0-flash": ModelConfig(
        name="gemini-2.0-flash",
        provider="gemini",
        max_tokens=2048,
        temperature=0.7,
        context_window=8192,
    ),
    "gemini-1.5-pro": ModelConfig(
        name="gemini-1.5-pro",
        provider="gemini",
        max_tokens=2048,
        temperature=0.7,
        context_window=32768,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelConfig object
        
    Raises:
        KeyError: If model name is not found
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Default fallback for unknown models (assume Ollama)
    return ModelConfig(
        name=model_name,
        provider="ollama",
        max_tokens=1024,
        temperature=0.7,
        context_window=4096,
    )


def list_available_models() -> dict[str, ModelConfig]:
    """List all available model configurations.
    
    Returns:
        Dictionary of model name to ModelConfig
    """
    return MODEL_CONFIGS.copy()


def create_model_for_adk(model_name: str) -> str:
    """Create a model configuration for ADK.
    
    ADK accepts model names as strings and handles the backend routing.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model name string for ADK
    """
    config = get_model_config(model_name)
    return config.name
