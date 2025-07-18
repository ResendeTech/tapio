# Tapio Assistant with Google ADK

This document explains how to use the new Google Agent Development Kit (ADK) implementation of Tapio Assistant, which provides a more robust, production-ready alternative to the Gradio interface.

## Overview

The ADK implementation provides:

- **Agent-First Architecture**: Built specifically for AI agents with conversation state management
- **Production-Ready APIs**: FastAPI-based backend with comprehensive API endpoints
- **Built-in Development UI**: Professional debugging and testing interface
- **Event-Driven Communication**: Proper agent communication and workflow management
- **Production Deployment**: Ready for containerization and cloud deployment

## Installation

Google ADK is already included in the project dependencies. If you need to install it separately:

```bash
uv add google-adk
```

## Usage

### Starting the ADK Server

Launch the Tapio Assistant using the ADK server:

```bash
# Basic usage
uv run -m tapio.cli adk-server

# With custom configuration
uv run -m tapio.cli adk-server --model-name llama3.2 --port 8000 --host 0.0.0.0

# Development mode with auto-reload
uv run -m tapio.cli adk-server --reload --web-ui

# Quick development server
uv run -m tapio.cli dev
```

### Available Endpoints

Once the server is running, you can access:

- **Web Development UI**: `http://localhost:8000` - Interactive agent testing interface
- **API Documentation**: `http://localhost:8000/docs` - Swagger/OpenAPI documentation
- **Health Check**: `http://localhost:8000/health` - Service health status
- **Service Info**: `http://localhost:8000/info` - Service configuration details

### API Usage

#### Chat with the Agent

Send a POST request to interact with the agent:

```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "agentName": "TapioAssistant",
    "input": "How do I apply for a residence permit in Finland?",
    "sessionId": "user123"
  }'
```

#### Streaming Responses

For real-time streaming responses:

```bash
curl -X POST "http://localhost:8000/run_sse" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "agentName": "TapioAssistant", 
    "input": "What documents do I need for family reunification?",
    "sessionId": "user123"
  }'
```

### Web Development UI Features

The built-in web interface provides:

1. **Interactive Chat**: Direct conversation with the Tapio Assistant
2. **Event Inspection**: View detailed execution traces, prompts, and tool calls
3. **Session Management**: Track conversation history and state
4. **Tool Monitoring**: See RAG document retrieval and LLM responses
5. **Performance Metrics**: Monitor response times and token usage

### Agent Configuration

The Tapio Assistant agent is configured with:

- **RAG Integration**: Automatic search of Finnish immigration documents
- **Contextual Responses**: Maintains conversation history and context
- **Source Attribution**: Provides document sources for transparency
- **Error Handling**: Graceful handling of failures with helpful messages

### Comparison: Gradio vs ADK

| Feature | Gradio | Google ADK |
|---------|--------|------------|
| **Interface** | Web UI only | API + Web UI + CLI |
| **Production Ready** | Development focused | Production ready |
| **Event Tracking** | Limited | Comprehensive |
| **Agent Communication** | Not supported | Built-in |
| **API Integration** | Basic | Full REST/SSE APIs |
| **Deployment** | Manual | Cloud-native |
| **Monitoring** | Basic | Advanced telemetry |
| **Multi-Agent** | Not supported | Native support |

### Migration from Gradio

If you're migrating from the Gradio implementation:

1. **Keep existing commands**: All crawl, parse, and vectorize commands work unchanged
2. **Use new server**: Replace `tapio-app` with `adk-server`
3. **Update integrations**: Use REST APIs instead of Gradio's interface
4. **Enhanced features**: Leverage event tracking and agent workflows

### Development Workflow

1. **Start Development Server**:
   ```bash
   uv run -m tapio.cli dev
   ```

2. **Access Web UI**: Open `http://localhost:8000` in your browser

3. **Test Queries**: Use the interactive interface to test various questions

4. **Monitor Events**: Use the Events tab to debug RAG retrieval and responses

5. **API Testing**: Use the `/docs` endpoint to test API endpoints

### Production Deployment

The ADK server is designed for production deployment:

```bash
# Production-ready server
uv run -m tapio.cli adk-server \
  --host 0.0.0.0 \
  --port 8000 \
  --no-web-ui
```

Or use the Google Cloud deployment tools:

```bash
# Deploy to Google Cloud Run (requires ADK CLI tools)
adk deploy --platform cloudrun
```

### Troubleshooting

**Common Issues:**

1. **Model Not Found**: Ensure Ollama is running and the model is pulled:
   ```bash
   ollama pull llama3.2
   ```

2. **ChromaDB Issues**: Ensure vectorization has been completed:
   ```bash
   uv run -m tapio.cli vectorize
   ```

3. **Port Conflicts**: Use a different port:
   ```bash
   uv run -m tapio.cli adk-server --port 8001
   ```

**Debug Mode:**

Enable detailed logging by setting the environment variable:

```bash
export LOGGING_LEVEL=DEBUG
uv run -m tapio.cli adk-server --reload
```

## Benefits of ADK Implementation

1. **Professional Development**: Built-in debugging and monitoring tools
2. **Production Ready**: Proper error handling, logging, and metrics
3. **Scalable Architecture**: Multi-agent workflows and communication
4. **API-First**: Full REST API with comprehensive documentation
5. **Cloud Native**: Easy deployment to Google Cloud and other platforms
6. **Event-Driven**: Comprehensive tracking of agent behavior and performance

The ADK implementation provides a solid foundation for building production-grade AI agent applications while maintaining all the RAG capabilities of the original Tapio Assistant.
