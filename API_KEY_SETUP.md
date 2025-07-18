# 🔑 Gemini API Key Setup Guide

## Quick Setup

### Option 1: Interactive Setup (Recommended)
```bash
uv run tapio setup-api-keys
```

### Option 2: Manual Setup

1. **Get your API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

2. **Set your API key** using one of these methods:

   **Method A: Environment Variable (Recommended)**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

   **Method B: .env File**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your API key
   echo "GEMINI_API_KEY=your_api_key_here" >> .env
   ```

   **Method C: Shell Profile (Permanent)**
   ```bash
   # Add to ~/.bashrc, ~/.zshrc, or ~/.profile
   echo 'export GEMINI_API_KEY="your_api_key_here"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Usage

### Start with Gemini (if API key is set)
```bash
uv run tapio adk-server --model-name gemini-2.0-flash
```

### Auto-detect model (uses Gemini if API key available, otherwise Ollama)
```bash
uv run tapio adk-server
```

### List available models
```bash
uv run tapio list-models
```

## Supported Models

**Gemini Models (require API key):**
- `gemini-2.0-flash` - Fast, efficient model
- `gemini-1.5-pro` - Advanced model with large context

**Ollama Models (local, no API key needed):**
- `llama3.2` - Meta's Llama 3.2
- `llama3.1` - Meta's Llama 3.1
- `mistral` - Mistral AI model

## Troubleshooting

**Error: "Gemini API key required"**
- Make sure you've set one of: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `GOOGLE_AI_API_KEY`
- Check that your API key is valid

**Error: "Model not found"**
- For Ollama models, make sure Ollama is running: `ollama serve`
- Pull the model: `ollama pull llama3.2`

**Error: "google-generativeai package not installed"**
- This should be automatically installed. If not: `uv add google-generativeai`
