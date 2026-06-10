"""Tests for the Orchestrator service (mocked retriever + LLM)."""

from unittest.mock import MagicMock

from chatbot.api.services.orchestrator import ChatResult, Orchestrator
from chatbot.llm_gateway.base import LLMResponse, TokenUsage
from chatbot.retriever.base import RetrievalResult


def _make_retrieval_results():
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            text="Landfills are engineered facilities for waste disposal.",
            score=0.92,
            book_title="Solid Waste Engineering",
            chapter="Chapter 5",
            page_number=120,
            page_range="120-122",
            chunk_id="chunk_1",
        ),
        RetrievalResult(
            text="Leachate management is critical for landfill operations.",
            score=0.85,
            book_title="Solid Waste Engineering",
            chapter="Chapter 6",
            page_number=145,
            page_range="145-147",
            chunk_id="chunk_2",
        ),
    ]


def _make_llm_response():
    """Create a sample LLM response."""
    return LLMResponse(
        answer="Landfills are engineered facilities designed for safe waste disposal. (Source: Solid Waste Engineering, Chapter 5, pages 120-122)",
        model="llama-3.1-70b-versatile",
        provider="groq",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=50),
    )


class TestOrchestrator:

    def test_chat_returns_chat_result(self):
        """chat() should return a ChatResult with answer, sources, and usage."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_results()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = _make_llm_response()

        orchestrator = Orchestrator(retriever=mock_retriever, llm=mock_llm)
        result = orchestrator.chat("What is a landfill?")

        assert isinstance(result, ChatResult)
        assert "landfill" in result.answer.lower()
        assert result.model == "llama-3.1-70b-versatile"
        assert result.provider == "groq"
        assert len(result.sources) == 2
        assert result.usage["total_tokens"] == 250

    def test_retriever_called_with_query(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_results()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _make_llm_response()

        orchestrator = Orchestrator(retriever=mock_retriever, llm=mock_llm)
        orchestrator.chat("How is leachate managed?", top_k=3)

        mock_retriever.retrieve.assert_called_once_with("How is leachate managed?", top_k=3)

    def test_llm_called_with_context(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_results()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _make_llm_response()

        orchestrator = Orchestrator(retriever=mock_retriever, llm=mock_llm)
        orchestrator.chat("test query")

        # LLM should be called with the query and a non-empty context string
        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert len(call_kwargs["context"]) > 0

    def test_sources_have_book_info(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_results()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _make_llm_response()

        orchestrator = Orchestrator(retriever=mock_retriever, llm=mock_llm)
        result = orchestrator.chat("What is waste management?")

        for source in result.sources:
            assert "book_title" in source
            assert "chapter" in source
            assert "relevance_score" in source

    def test_empty_retrieval_still_works(self):
        """Even with no chunks, orchestrator should return an answer."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_llm = MagicMock()
        mock_llm.generate.return_value = LLMResponse(
            answer="I don't have enough information.",
            model="test",
            provider="test",
            usage=TokenUsage(prompt_tokens=50, completion_tokens=10),
        )

        orchestrator = Orchestrator(retriever=mock_retriever, llm=mock_llm)
        result = orchestrator.chat("Unknown question about something obscure")

        assert result.answer == "I don't have enough information."
        assert result.sources == []

    def test_casual_message_skips_retrieval(self):
        """Greetings like 'hi' should skip RAG and return no sources."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = LLMResponse(
            answer="Hello! I'm the Waste Management Assistant.",
            model="test",
            provider="test",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10),
        )

        orchestrator = Orchestrator(retriever=mock_retriever, llm=mock_llm)
        result = orchestrator.chat("hi")

        # Retriever should NOT be called for greetings
        mock_retriever.retrieve.assert_not_called()
        assert result.sources == []
        assert len(result.answer) > 0

    def test_knowledge_question_uses_retrieval(self):
        """Knowledge questions should go through the full RAG pipeline."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_retrieval_results()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = _make_llm_response()

        orchestrator = Orchestrator(retriever=mock_retriever, llm=mock_llm)
        result = orchestrator.chat("How does leachate treatment work?")

        # Retriever SHOULD be called for knowledge questions
        mock_retriever.retrieve.assert_called_once()
        assert len(result.sources) > 0


class TestCasualDetection:
    """Test the casual message detection logic."""

    def test_greetings_are_casual(self):
        assert Orchestrator._is_casual_message("hi") is True
        assert Orchestrator._is_casual_message("Hello") is True
        assert Orchestrator._is_casual_message("hey there") is True
        assert Orchestrator._is_casual_message("Good morning") is True

    def test_thanks_are_casual(self):
        assert Orchestrator._is_casual_message("thanks") is True
        assert Orchestrator._is_casual_message("Thank you") is True

    def test_knowledge_questions_are_not_casual(self):
        assert Orchestrator._is_casual_message("What is leachate?") is False
        assert Orchestrator._is_casual_message("How do landfills work?") is False
        assert Orchestrator._is_casual_message("Explain the waste hierarchy") is False

    def test_short_alpha_is_casual(self):
        assert Orchestrator._is_casual_message("ok") is True
        assert Orchestrator._is_casual_message("hey") is True

