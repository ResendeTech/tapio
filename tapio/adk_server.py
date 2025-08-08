"""ADK-based server implementation for Tapio Assistant."""

import logging
import os
from pathlib import Path
from typing import Optional

from google.adk.cli.fast_api import get_fast_api_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TapioADKServer:
    """ADK-based server for the Tapio Assistant."""

    def __init__(
        self,
        agents_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        enable_web_ui: bool = True,
    ) -> None:
        """Initialize the Tapio ADK server.

        Args:
            agents_dir: Directory containing agent definitions
            host: Host to bind the server to
            port: Port to bind the server to
            enable_web_ui: Whether to enable the web development UI
        """
        self.host = host
        self.port = port
        self.enable_web_ui = enable_web_ui

        # Set agents directory - default to the tapio agents directory
        if agents_dir is None:
            current_dir = Path(__file__).parent
            self.agents_dir = str(current_dir / "agents")
        else:
            self.agents_dir = agents_dir

        logger.info(f"Tapio ADK Server initialized with agents from: {self.agents_dir}")

    def create_app(self):
        """Create and configure the FastAPI application."""
        try:
            # Create the main ADK FastAPI app
            app = get_fast_api_app(
                agents_dir=self.agents_dir,
                web=self.enable_web_ui,  # Enable the built-in web development UI
            )

            # Add custom health check endpoint
            @app.get("/health")
            async def health_check():
                """Health check endpoint for monitoring."""
                return {"status": "healthy", "service": "tapio-assistant"}

            # Add custom info endpoint
            @app.get("/info")
            async def service_info():
                """Service information endpoint."""
                return {
                    "service": "Tapio Assistant",
                    "description": "RAG-powered Finnish immigration assistant",
                    "version": "1.0.0",
                    "agents_dir": self.agents_dir,
                    "web_ui_enabled": self.enable_web_ui,
                }

            logger.info("ADK FastAPI application created successfully")
            return app

        except Exception as e:
            logger.error(f"Failed to create ADK app: {e}")
            raise

    def run(self, reload: bool = False) -> None:
        """Run the ADK server.

        Args:
            reload: Whether to enable auto-reload for development
        """
        try:
            import uvicorn

            app = self.create_app()

            logger.info(f"Starting Tapio ADK Server on {self.host}:{self.port}")
            if self.enable_web_ui:
                logger.info(f"Web development UI available at: http://{self.host}:{self.port}")
            logger.info(f"API documentation at: http://{self.host}:{self.port}/docs")

            # Run the server
            uvicorn.run(
                app,
                host=self.host,
                port=self.port,
                reload=reload,
                log_level="info",
            )

        except Exception as e:
            logger.error(f"Failed to start ADK server: {e}")
            raise


def launch_tapio_adk_server(
    agents_dir: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_web_ui: bool = True,
    reload: bool = False,
) -> None:
    """Launch the Tapio ADK server.

    Args:
        agents_dir: Directory containing agent definitions
        host: Host to bind the server to
        port: Port to bind the server to
        enable_web_ui: Whether to enable the web development UI
        reload: Whether to enable auto-reload for development
    """
    server = TapioADKServer(
        agents_dir=agents_dir,
        host=host,
        port=port,
        enable_web_ui=enable_web_ui,
    )
    server.run(reload=reload)


if __name__ == "__main__":
    # For development/testing
    launch_tapio_adk_server(reload=True)
