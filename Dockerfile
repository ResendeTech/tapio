# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy project files
COPY . .

# Install dependencies using UV
RUN uv sync

# Install Google ADK
RUN uv add google-adk

# Expose ports
EXPOSE 8000 8001 11434

# Set up environment
ENV PYTHONPATH="/workspace"
ENV TAPIO_DEFAULT_MODEL="llama3.2"

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start Ollama in background\n\
ollama serve &\n\
sleep 5\n\
# Pull the default model if it doesn'\''t exist\n\
ollama list | grep -q llama3.2 || ollama pull llama3.2\n\
# Start the ADK server\n\
exec uv run tapio adk-server --host 0.0.0.0 --port 8000 "$@"\n\
' > /workspace/start-adk.sh && chmod +x /workspace/start-adk.sh

# Default command
CMD ["/workspace/start-adk.sh"]
