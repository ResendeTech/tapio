#!/bin/bash
# Tapio ADK Setup Script for Dev Container

echo "🚀 Setting up Tapio with ADK in Dev Container..."

# Ensure we're in the right directory
cd /workspaces/tapio || cd /workspace || cd $(pwd)

echo "📦 Installing dependencies..."
uv sync --dev

echo "🔍 Checking ADK installation..."
uv run python -c "import google.adk; print('✅ Google ADK installed successfully')" || {
    echo "❌ ADK not found, installing..."
    uv add google-adk
}

echo "🤖 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found in PATH, but it should be available via devcontainer feature"
fi

echo "🏗️  Setting up Tapio ADK Agent..."
export TAPIO_DEFAULT_MODEL="llama3.2"
export TAPIO_CHROMA_DIR="chroma_db"

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick commands to try:"
echo "  uv run tapio --help"
echo "  uv run tapio adk-server --help"
echo "  uv run tapio list-sites"
echo ""
echo "🚀 To start the ADK server:"
echo "  uv run tapio adk-server --port 8000"
echo ""
