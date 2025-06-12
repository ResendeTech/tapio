"""Tests for the RAG orchestrator service."""

from unittest import mock

import pytest

from tapio.services.rag_orchestrator import RAGOrchestrator


@pytest.fixture
def rag_orchestrator():
    """Create a RAG orchestrator with mocked dependencies for testing."""
    # Mock the DocumentRetrievalService and LLMService
    with mock.patch("tapio.services.rag_orchestrator.DocumentRetrievalService") as mock_doc_service:
        with mock.patch("tapio.services.rag_orchestrator.LLMService") as mock_llm:
            # Configure mock instance behavior
            mock_doc_service_instance = mock_doc_service.return_value
            mock_llm_instance = mock_llm.return_value

            # Create RAGOrchestrator with mocked dependencies
            orchestrator = RAGOrchestrator(
                collection_name="test_collection",
                persist_directory="test_db",
                model_name="test_model",
            )

            # Set mocked properties for access in tests
            orchestrator.mock_doc_service = mock_doc_service_instance
            orchestrator.mock_llm_service = mock_llm_instance

            yield orchestrator


def test_rag_orchestrator_query(rag_orchestrator):
    """Test that RAG orchestrator correctly coordinates document retrieval and LLM generation."""
    # Mock the prompt loading functions
    with mock.patch("tapio.services.rag_orchestrator.load_prompt") as mock_load_prompt:
        # Configure mock return values
        mock_load_prompt.side_effect = [
            "Mocked system prompt",  # For system_prompt
            "Mocked user prompt with context",  # For user_query
        ]

        # Mock document retrieval
        mock_doc = mock.MagicMock()
        mock_doc.page_content = "Test document content"
        rag_orchestrator.mock_doc_service.retrieve_documents.return_value = [
            mock_doc,
        ]
        rag_orchestrator.mock_doc_service.format_documents_as_context.return_value = "Test document content"

        # Mock LLM response
        rag_orchestrator.mock_llm_service.generate_response.return_value = "Test LLM response"

        # Call the method under test
        response, docs = rag_orchestrator.query("Test query")

        # Verify document retrieval was called
        rag_orchestrator.mock_doc_service.retrieve_documents.assert_called_once_with(
            "Test query",
        )
        rag_orchestrator.mock_doc_service.format_documents_as_context.assert_called_once_with(
            [
                mock_doc,
            ],
        )

        # Check that the prompts were loaded
        assert mock_load_prompt.call_count == 2
        mock_load_prompt.assert_any_call("system_prompt")

        # Check that the user prompt was loaded with the correct variables
        user_prompt_calls = [call for call in mock_load_prompt.call_args_list if call[0][0] == "user_query"]
        assert len(user_prompt_calls) == 1

        # Get the kwargs passed to load_prompt for user_query
        user_prompt_kwargs = user_prompt_calls[0][1]
        assert "context" in user_prompt_kwargs
        assert "question" in user_prompt_kwargs
        assert user_prompt_kwargs["question"] == "Test query"
        assert user_prompt_kwargs["context"] == "Test document content"

        # Verify the LLM was called with the correct prompts
        rag_orchestrator.mock_llm_service.generate_response.assert_called_once()
        call_args = rag_orchestrator.mock_llm_service.generate_response.call_args[1]
        assert call_args["system_prompt"] == "Mocked system prompt"
        assert call_args["prompt"] == "Mocked user prompt with context"

        # Verify the results
        assert response == "Test LLM response"
        assert docs == [mock_doc]


def test_rag_orchestrator_query_stream(rag_orchestrator):
    """Test that RAG orchestrator correctly coordinates streaming response."""
    # Mock the prompt loading functions
    with mock.patch("tapio.services.rag_orchestrator.load_prompt") as mock_load_prompt:
        # Configure mock return values
        mock_load_prompt.side_effect = [
            "Mocked system prompt",  # For system_prompt
            "Mocked user prompt with context",  # For user_query
        ]

        # Mock document retrieval
        mock_doc = mock.MagicMock()
        mock_doc.page_content = "Test document content"
        rag_orchestrator.mock_doc_service.retrieve_documents.return_value = [
            mock_doc,
        ]
        rag_orchestrator.mock_doc_service.format_documents_as_context.return_value = "Test document content"

        # Mock LLM streaming response
        def mock_stream():
            yield "Test "
            yield "streaming "
            yield "response"

        rag_orchestrator.mock_llm_service.generate_response_stream.return_value = mock_stream()

        # Call the method under test
        response_stream, docs = rag_orchestrator.query_stream("Test query")

        # Verify document retrieval was called (this happens immediately)
        rag_orchestrator.mock_doc_service.retrieve_documents.assert_called_once_with(
            "Test query",
        )
        rag_orchestrator.mock_doc_service.format_documents_as_context.assert_called_once_with(
            [
                mock_doc,
            ],
        )

        # Verify the results
        assert docs == [mock_doc]

        # Collect all chunks from the stream (this triggers the LLM call)
        chunks = list(response_stream)
        assert chunks == ["Test ", "streaming ", "response"]

        # Verify the LLM was called with streaming (check after consuming the generator)
        rag_orchestrator.mock_llm_service.generate_response_stream.assert_called_once()


def test_rag_orchestrator_check_model_availability(rag_orchestrator):
    """Test that RAG orchestrator correctly checks model availability."""
    # Mock LLM model availability check
    rag_orchestrator.mock_llm_service.check_model_availability.return_value = True

    # Call the method under test
    result = rag_orchestrator.check_model_availability()

    # Verify the LLM service was called
    rag_orchestrator.mock_llm_service.check_model_availability.assert_called_once()

    # Verify the result
    assert result is True


def test_rag_orchestrator_format_documents_for_display(rag_orchestrator):
    """Test that RAG orchestrator correctly delegates document formatting."""
    # Mock documents
    mock_docs = [mock.MagicMock(), mock.MagicMock()]

    # Mock document service formatting
    rag_orchestrator.mock_doc_service.format_documents_for_display.return_value = "Formatted documents"

    # Call the method under test
    result = rag_orchestrator.format_documents_for_display(mock_docs)

    # Verify the document service was called
    rag_orchestrator.mock_doc_service.format_documents_for_display.assert_called_once_with(
        mock_docs,
    )

    # Verify the result
    assert result == "Formatted documents"
