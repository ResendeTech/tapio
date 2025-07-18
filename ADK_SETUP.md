# Tapio ADK Development Guide

## Opening in Dev Container

To resolve the import issues and work with the full ADK implementation, follow these steps:

### Method 1: VS Code Dev Container (Recommended)

1. **Open in VS Code**:
   ```bash
   code .
   ```

2. **Reopen in Container**:
   - VS Code will detect the `.devcontainer/devcontainer.json` file
   - Click "Reopen in Container" when prompted
   - Or use Command Palette (`Ctrl+Shift+P`) → "Dev Containers: Reopen in Container"

3. **Wait for Setup**:
   - The container will automatically install all dependencies including Google ADK
   - This includes Python 3.12, uv, Ollama, and all required packages

### Method 2: GitHub Codespaces

1. **Create Codespace**:
   - Go to your GitHub repository
   - Click "Code" → "Codespaces" → "Create codespace on main"
   - Or use: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ResendeTech/tapio?quickstart=1)

### Method 3: Manual Container Setup

If you prefer to run the container manually:

```bash
# Build and run the dev container
docker build -t tapio-adk -f .devcontainer/Dockerfile .
docker run -it -p 8000:8000 -p 8001:8001 -p 11434:11434 -v $(pwd):/workspace tapio-adk
```

## After Container Setup

Once in the container, run the setup script:

```bash
chmod +x setup-adk.sh
./setup-adk.sh
```

## Verifying the Setup

Test that all imports work:

```bash
# Test ADK imports
uv run python -c "
from google.adk.agents import LlmAgent
from google.adk.models import Gemini  
from google.adk.tools import FunctionTool
from google.adk.cli.fast_api import get_fast_api_app
print('✅ All ADK imports successful!')
"

# Test Tapio agent creation
uv run python -c "
import sys
sys.path.append('.')
from tapio.agents.tapio_assistant.agent import create_tapio_agent
agent = create_tapio_agent()
print(f'✅ Tapio agent created: {agent.name}')
"
```

## Starting the ADK Server

```bash
# Start the ADK development server
uv run tapio adk-server --port 8000 --host 0.0.0.0

# Or with auto-reload for development
uv run tapio adk-server --port 8000 --host 0.0.0.0 --reload
```

The server will be available at:
- **Web Development UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Agent API**: http://localhost:8000/api/agents

## Key Differences from Gradio

| Feature | Gradio | ADK |
|---------|--------|-----|
| Interface | Custom web UI | Professional development UI + API |
| Architecture | Monolithic | Agent-based with tools |
| Deployment | Basic | Production-ready (FastAPI) |
| Development | Manual testing | Built-in testing & evaluation |
| Extensibility | Limited | Modular agent system |
| State Management | Session-based | Proper conversation state |

## Troubleshooting

### Import Issues
- Make sure you're in the dev container
- Run `uv sync --dev` to ensure all dependencies are installed
- Check Python path: `uv run python -c "import sys; print(sys.path)"`

### Ollama Issues
- Ollama should be available in the container via the devcontainer feature
- Check: `ollama list` to see installed models
- Install model: `ollama pull llama3.2`

### Port Issues
- The devcontainer forwards ports 8000, 8001, and 11434 automatically
- Make sure no other services are using these ports

## Migration Benefits

✅ **Professional Development Experience**
- Built-in web UI for testing and debugging
- Comprehensive API documentation
- Event tracing and debugging tools

✅ **Production Ready**
- FastAPI backend with automatic OpenAPI docs
- Proper error handling and logging
- Containerization and deployment support

✅ **Agent Architecture**
- Modular design with reusable tools
- Easy to extend with new capabilities
- Proper conversation state management

✅ **Testing & Evaluation**
- Built-in evaluation framework
- Agent testing tools
- Performance monitoring
