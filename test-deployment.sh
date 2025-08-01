#!/bin/bash

echo "🚀 Testing Tapio Deployment Locally"
echo "=================================="

# Check if API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not found"
    echo "Please set your API key first:"
    echo "export GEMINI_API_KEY='your-key-here'"
    echo ""
    echo "Or create a .env file with:"
    echo "GEMINI_API_KEY=your-key-here"
    exit 1
fi

echo "✅ API key found"

# Test Docker build
echo ""
echo "🐳 Testing Docker build..."
if docker build -t tapio-test . > /dev/null 2>&1; then
    echo "✅ Docker build successful"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Test container run
echo ""
echo "🔄 Starting container (will run for 30 seconds)..."
docker run -d --name tapio-test-container \
    -p 8080:8080 \
    -e GEMINI_API_KEY="$GEMINI_API_KEY" \
    -e TAPIO_DEFAULT_MODEL="gemini-2.0-flash" \
    tapio-test

# Wait for startup
sleep 10

# Test health endpoint
echo ""
echo "🩺 Testing health endpoint..."
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
fi

# Test API docs
echo ""
echo "📚 Testing API docs..."
if curl -s http://localhost:8080/docs > /dev/null; then
    echo "✅ API docs accessible"
else
    echo "❌ API docs not accessible"
fi

echo ""
echo "🌐 Your service is running at:"
echo "   Health: http://localhost:8080/health"
echo "   API Docs: http://localhost:8080/docs"
echo "   API: http://localhost:8080/api/chat"

echo ""
echo "🧪 Testing chat API..."
response=$(curl -s -X POST "http://localhost:8080/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, what is Tapio?"}' | jq -r '.response' 2>/dev/null)

if [ ! -z "$response" ] && [ "$response" != "null" ]; then
    echo "✅ Chat API working"
    echo "Response: $response"
else
    echo "❌ Chat API test failed"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."
docker stop tapio-test-container > /dev/null 2>&1
docker rm tapio-test-container > /dev/null 2>&1
docker rmi tapio-test > /dev/null 2>&1

echo ""
echo "✅ Local deployment test complete!"
echo ""
echo "Ready for cloud deployment? Choose your platform:"
echo "  - Railway: https://railway.app (fastest)"
echo "  - Render: https://render.com (free tier)"
echo "  - Google Cloud Run: gcloud run deploy"
echo "  - Hugging Face: https://huggingface.co/spaces"
