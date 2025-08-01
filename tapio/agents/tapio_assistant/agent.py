"""Tapio Assistant ADK Agent - RAG-powered Finnish immigration assistant."""

import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from tapio.services.rag_orchestrator import RAGOrchestrator

# Default constants for agent configuration
DEFAULT_CHROMA_COLLECTION = "migri_docs"
DEFAULT_CHROMA_DB_PATH = "chroma_db"
DEFAULT_MODEL_NAME = "gemini-2.0-flash"  # Default to Gemini since we have API key support
DEFAULT_MAX_TOKENS = 1024
DEFAULT_NUM_RESULTS = 5

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TapioRAGTool:
    """Tool for retrieving information from Finnish immigration documents using RAG."""

    def __init__(
        self,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        persist_directory: str = DEFAULT_CHROMA_DB_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        num_results: int = DEFAULT_NUM_RESULTS,
    ) -> None:
        """Initialize the RAG tool.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory where the ChromaDB database is stored
            model_name: Name of the LLM model to use
            max_tokens: Maximum number of tokens to generate
            num_results: Number of documents to retrieve from the vector store
        """
        self.rag_orchestrator = RAGOrchestrator(
            collection_name=collection_name,
            persist_directory=persist_directory,
            model_name=model_name,
            max_tokens=max_tokens,
            num_results=num_results,
        )

    def search_immigration_docs(self, query: str) -> str:
        """Search Finnish immigration documents for relevant information.

        Args:
            query: The user's question about Finnish immigration

        Returns:
            Relevant information and guidance from official Finnish immigration sources
        """
        try:
            logger.info(f"Processing RAG query: {query}")

            # Use the existing RAG orchestrator to get response and documents
            response, retrieved_docs = self.rag_orchestrator.query(
                query_text=query,
                history=None,
            )

            # Format the response with source information
            formatted_docs = self.rag_orchestrator.format_documents_for_display(
                retrieved_docs,
            )

            # Combine response with sources for transparency
            full_response = f"{response}\n\n**Sources:**\n{formatted_docs}"

            return full_response

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return (
                "I encountered an error while searching for information. "
                "Please try rephrasing your question or contact support."
            )


def create_tapio_agent(
    model_name: str = None,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    persist_directory: str = DEFAULT_CHROMA_DB_PATH,
) -> LlmAgent:
    """Create and configure the Tapio Assistant agent.

    Args:
        model_name: Name of the LLM model to use (defaults to env var or llama3.2)
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory where the ChromaDB database is stored

    Returns:
        Configured Tapio Assistant agent
    """
    # Use environment variable or auto-detect based on available API keys
    if model_name is None:
        from tapio.config.settings import get_default_model

        model_name = get_default_model()

    logger.info(f"Creating Tapio agent with model: {model_name}")

    # Initialize the RAG tool
    rag_tool = TapioRAGTool(
        collection_name=collection_name,
        persist_directory=persist_directory,
        model_name=model_name,
    )

    # Create function tool for RAG search
    search_tool = FunctionTool(func=rag_tool.search_immigration_docs)

    # Create the agent with comprehensive instructions
    # ADK accepts model name as string and handles the backend automatically
    agent = LlmAgent(
        name="TapioAssistant",
        description=(
            "A helpful assistant specialized in Finnish immigration, "
            "residence permits, work permits, family reunification, and citizenship."
        ),
        model=model_name,  # ADK accepts model name as string
        instruction="""You are Tapio, a helpful assistant specializing in Finnish immigration processes. 
        You help EU and non-EU citizens navigate Finnish immigration, including:
        
        - Residence permits and applications
        - Work permits and employment requirements  
        - Family reunification procedures
        - Student visas and study in Finland
        - Finnish citizenship requirements
        - Asylum and refugee procedures
        - Rights and obligations in Finland
        
        **Important Guidelines:**
        1. Always use the search_immigration_docs tool to find current, accurate information
        2. Provide step-by-step guidance when possible
        3. Include relevant document requirements and deadlines
        4. Mention that users should verify information with official sources at migri.fi
        5. Be empathetic - immigration can be stressful and complex
        6. If unsure about something, recommend contacting Migri directly
        
        **Response Format:**
        - Give clear, actionable answers
        - Include document requirements
        - Mention processing times when relevant
        - Always include disclaimer about verifying with official sources
        """,
        tools=[search_tool],
    )

    return agent


# Create the root agent that ADK will discover
root_agent = create_tapio_agent()
