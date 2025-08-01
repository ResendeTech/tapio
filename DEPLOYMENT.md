# 🚀 Deployment Guide for Tapio Chat Service

## Quick Deployment Options

### 1. Railway (Fastest - 5 minutes)

**Step 1:** Push your code to GitHub
```bash
git add .
git commit -m "Add deployment configs"
git push origin main
```

**Step 2:** Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `tapio` repository
4. Railway will auto-detect the `railway.toml` configuration

**Step 3:** Set Environment Variables
In Railway dashboard:
- Add `GEMINI_API_KEY` with your API key
- Add `TAPIO_DEFAULT_MODEL=gemini-2.0-flash`

**Step 4:** Access your chat
- Railway will provide a URL like `https://tapio-production.up.railway.app`
- Visit `https://your-url.railway.app` to use the chat interface

### 2. Render (Free tier available)

**Step 1:** Create `render.yaml`
```yaml
services:
  - type: web
    name: tapio-chat
    env: python
    buildCommand: "uv sync && uv pip install -e ."
    startCommand: "uv run tapio adk-server --host 0.0.0.0 --port $PORT"
    envVars:
      - key: GEMINI_API_KEY
        sync: false
      - key: TAPIO_DEFAULT_MODEL
        value: gemini-2.0-flash
```

**Step 2:** Deploy
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Set environment variables in Render dashboard

### 3. Google Cloud Run (Scalable)

**Step 1:** Build and push Docker image
```bash
# Build the image
docker build -t gcr.io/YOUR-PROJECT-ID/tapio-chat .

# Push to Google Container Registry
docker push gcr.io/YOUR-PROJECT-ID/tapio-chat
```

**Step 2:** Deploy to Cloud Run
```bash
gcloud run deploy tapio-chat \
  --image gcr.io/YOUR-PROJECT-ID/tapio-chat \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY,TAPIO_DEFAULT_MODEL=gemini-2.0-flash
```

### 4. Docker (Self-hosted)

**Option A: With Gemini API (Recommended)**
```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Run with docker-compose
docker-compose up tapio-chat
```

**Option B: With local Ollama**
```bash
# Run local version with Ollama
docker-compose --profile local up tapio-local
```

### 5. Hugging Face Spaces (Free)

Create `app.py`:
```python
import os
import gradio as gr
from tapio.services.rag_orchestrator import RAGOrchestrator

# Set API key from Hugging Face secrets
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

def chat_interface(message, history):
    rag = RAGOrchestrator(model_name="gemini-2.0-flash")
    response, _ = rag.query(message, history)
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="Tapio - Finnish Immigration Assistant",
    description="Ask questions about Finnish immigration, visas, and residence permits."
)

if __name__ == "__main__":
    demo.launch()
```

## Environment Variables Required

For all deployments, you need:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
TAPIO_DEFAULT_MODEL=gemini-2.0-flash
```

## Testing Your Deployment

1. **Health Check**: Visit `https://your-url/health`
2. **API Docs**: Visit `https://your-url/docs`
3. **Chat Interface**: Visit `https://your-url/` (if web UI is enabled)

## Cost Estimates

| Platform | Cost | Features |
|----------|------|----------|
| Railway | ~$5/month | Easy setup, auto-scaling |
| Render | Free tier available | Good for testing |
| Google Cloud Run | Pay-per-use | Highly scalable |
| Hugging Face | Free | Great for demos |
| Self-hosted | Server costs only | Full control |

## Recommended: Railway for Quick Testing

For the fastest deployment to test online:

1. **Push to GitHub**
2. **Connect to Railway** 
3. **Set API key**
4. **Share the URL**

Your chat service will be live in about 5 minutes!

## Security Considerations

- Never commit API keys to git
- Use environment variables for secrets
- Enable HTTPS (automatic on most platforms)
- Consider rate limiting for production use
- Monitor API usage and costs
