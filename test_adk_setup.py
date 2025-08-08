#!/usr/bin/env python3
"""Test script to verify Tapio ADK setup."""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic Python imports
        import os
        import logging
        print("✅ Basic Python imports")
        
        # Test Tapio imports
        from tapio.services.rag_orchestrator import RAGOrchestrator
        print("✅ Tapio core imports")
        
        # Test ADK imports
        from google.adk.agents import LlmAgent
        from google.adk.models import Gemini
        from google.adk.tools import FunctionTool
        from google.adk.cli.fast_api import get_fast_api_app
        print("✅ Google ADK imports")
        
        # Test uvicorn import
        import uvicorn
        print("✅ Uvicorn import")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test creating the Tapio agent."""
    print("\n🤖 Testing agent creation...")
    
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from tapio.agents.tapio_assistant.agent import create_tapio_agent
        
        # Create agent with test parameters
        agent = create_tapio_agent(
            model_name="llama3.2",
            collection_name="test_collection",
            persist_directory="test_chroma_db",
        )
        
        print(f"✅ Agent created successfully: {agent.name}")
        print(f"   Model: {agent.model}")
        print(f"   Tools: {len(agent.tools)} tools available")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        traceback.print_exc()
        return False

def test_adk_server():
    """Test ADK server creation."""
    print("\n🌐 Testing ADK server setup...")
    
    try:
        from tapio.adk_server import TapioADKServer
        
        # Create server instance (don't start it)
        server = TapioADKServer(
            host="127.0.0.1",
            port=8000,
            enable_web_ui=True,
        )
        
        print("✅ ADK server instance created")
        
        # Test app creation
        app = server.create_app()
        print("✅ FastAPI app created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ADK server test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Tapio ADK Setup Verification")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_agent_creation,
        test_adk_server,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Tapio ADK setup is ready.")
        print("\n🚀 Next steps:")
        print("   1. Ensure you have vector data: uv run tapio vectorize")
        print("   2. Start the server: uv run tapio adk-server")
        print("   3. Open http://localhost:8000 in your browser")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
