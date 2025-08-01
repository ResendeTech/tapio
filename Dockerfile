# Use Python 3.12 slim image for production deployment
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy application code
COPY . .

# Install the project in editable mode
RUN uv pip install -e .

# Create non-root user for security
RUN useradd -m -u 1000 tapio && chown -R tapio:tapio /app
USER tapio

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "tapio", "adk-server", "--host", "0.0.0.0", "--port", "8000"]

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
