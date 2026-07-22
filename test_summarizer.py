"""
test_summarizer.py

Unit tests for the news summarizer pipeline. Every external call
(NewsAPI, OpenAI, Cohere) is mocked — these tests never hit the network
or spend real API budget, so they're safe to run anytime, without keys.

Run: pytest test_summarizer.py -v
"""

from unittest.mock import Mock, patch

import pytest

import config
import llm_providers as lp
import news_api
from news_api import NewsAPIClient, NewsAPIError
from summarizer import NewsSummarizer


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Prevent retry/backoff logic from actually sleeping during tests."""
    monkeypatch.setattr(news_api.time, "sleep", lambda *_: None)
    monkeypatch.setattr(lp.time, "sleep", lambda *_: None)


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------
class TestConfig:
    def test_looks_like_placeholder_detects_empty_and_placeholder_values(self):
        assert config._looks_like_placeholder("") is True
        assert config._looks_like_placeholder(None) is True
        assert config._looks_like_placeholder("<your key here>") is True
        assert config._looks_like_placeholder("your_api_key") is True
        assert config._looks_like_placeholder("sk-real-looking-key-123") is False

    def test_validate_config_reports_missing_keys(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "REQUIRED_KEYS",
            {"OPENAI_API_KEY": None, "COHERE_API_KEY": "co-real-key", "NEWS_API_KEY": ""},
        )
        is_valid, missing = config.validate_config()
        assert is_valid is False
        assert set(missing) == {"OPENAI_API_KEY", "NEWS_API_KEY"}

    def test_validate_config_passes_when_all_keys_set(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "REQUIRED_KEYS",
            {"OPENAI_API_KEY": "sk-1", "COHERE_API_KEY": "co-1", "NEWS_API_KEY": "news-1"},
        )
        is_valid, missing = config.validate_config()
        assert is_valid is True
        assert missing == []


# ---------------------------------------------------------------------------
# news_api.py
# ---------------------------------------------------------------------------
class TestNewsAPIClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(news_api.config, "NEWS_API_KEY", None)
        with pytest.raises(NewsAPIError):
            NewsAPIClient(api_key=None)

    def test_get_top_headlines_returns_articles(self):
        client = NewsAPIClient(api_key="fake-key", timeout=5, max_retries=1)
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {
            "status": "ok",
            "articles": [{"title": "A"}, {"title": "B"}],
        }
        with patch.object(news_api.requests, "get", return_value=fake_response) as mock_get:
            articles = client.get_top_headlines(category="technology", page_size=2)

        assert len(articles) == 2
        assert mock_get.call_args.kwargs["params"]["category"] == "technology"
        assert mock_get.call_args.kwargs["params"]["pageSize"] == 2

    def test_get_top_headlines_raises_on_api_error_status(self):
        client = NewsAPIClient(api_key="fake-key", timeout=5, max_retries=1)
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {
            "status": "error",
            "code": "apiKeyInvalid",
            "message": "bad key",
        }
        with patch.object(news_api.requests, "get", return_value=fake_response):
            with pytest.raises(NewsAPIError):
                client.get_top_headlines()

    def test_retries_on_429_then_succeeds(self):
        client = NewsAPIClient(api_key="fake-key", timeout=5, max_retries=2)

        rate_limited = Mock(status_code=429)
        success = Mock(status_code=200)
        success.raise_for_status = lambda: None
        success.json = lambda: {"status": "ok", "articles": [{"title": "A"}]}

        with patch.object(news_api.requests, "get", side_effect=[rate_limited, success]):
            articles = client.get_top_headlines()

        assert len(articles) == 1


# ---------------------------------------------------------------------------
# llm_providers.py
# ---------------------------------------------------------------------------
class TestTokenBudgetManager:
    def test_calculate_cost_uses_provider_pricing(self):
        manager = lp.TokenBudgetManager(daily_budget=5.0)
        cost = manager.calculate_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.15 + 0.60)

    def test_calculate_cost_unknown_model_returns_zero(self):
        manager = lp.TokenBudgetManager(daily_budget=5.0)
        assert manager.calculate_cost("openai", "not-a-real-model", 1000, 1000) == 0.0

    def test_track_request_raises_when_over_budget(self):
        manager = lp.TokenBudgetManager(daily_budget=0.00001)
        with pytest.raises(lp.BudgetExceededError):
            manager.track_request("openai", "gpt-4o-mini", 100_000, 100_000)

    def test_track_request_accumulates_usage(self):
        manager = lp.TokenBudgetManager(daily_budget=5.0)
        manager.track_request("openai", "gpt-4o-mini", 100, 50)
        assert manager.provider_usage["openai"]["input_tokens"] == 100
        assert manager.provider_usage["openai"]["output_tokens"] == 50
        assert manager.used_budget > 0


class TestProviderCalls:
    def test_ask_openai_returns_response_and_usage(self):
        fake_response = Mock()
        fake_response.choices = [Mock(message=Mock(content="hello"))]
        fake_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        with patch.object(lp.openai_client.chat.completions, "create", return_value=fake_response):
            result = lp.ask_openai("test prompt")

        assert result["response"] == "hello"
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    def test_ask_cohere_returns_response_and_usage(self):
        block = Mock(type="text", text="Positive")
        fake_response = Mock()
        fake_response.message = Mock(content=[block])
        fake_response.usage = Mock(billed_units=Mock(input_tokens=8, output_tokens=2))

        with patch.object(lp.cohere_client, "chat", return_value=fake_response):
            result = lp.ask_cohere("test prompt")

        assert result["response"] == "Positive"
        assert result["usage"] == {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def test_ask_with_fallback_uses_secondary_on_primary_failure(self):
        block = Mock(type="text", text="fallback response")
        cohere_response = Mock()
        cohere_response.message = Mock(content=[block])
        cohere_response.usage = Mock(billed_units=Mock(input_tokens=5, output_tokens=3))

        with patch.object(lp.openai_client.chat.completions, "create", side_effect=Exception("down")), \
             patch.object(lp.cohere_client, "chat", return_value=cohere_response):
            result = lp.ask_with_fallback("test prompt", primary="openai", secondary="cohere")

        assert result["provider"] == "cohere"
        assert result["response"] == "fallback response"


# ---------------------------------------------------------------------------
# summarizer.py
# ---------------------------------------------------------------------------
class TestNewsSummarizer:
    @pytest.fixture
    def openai_response(self):
        response = Mock()
        response.choices = [Mock(message=Mock(content="A concise summary."))]
        response.usage = Mock(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        return response

    @pytest.fixture
    def cohere_response(self):
        block = Mock(type="text", text="Neutral")
        response = Mock()
        response.message = Mock(content=[block])
        response.usage = Mock(billed_units=Mock(input_tokens=12, output_tokens=4))
        return response

    def test_process_articles_summarizes_and_scores_sentiment(self, openai_response, cohere_response):
        fake_articles = [
            {"title": "Article One", "description": "Details one.", "url": "http://example.com/1"},
            {"title": "Article Two", "description": "Details two.", "url": "http://example.com/2"},
        ]

        with patch.object(NewsAPIClient, "get_top_headlines", return_value=fake_articles), \
             patch.object(lp.openai_client.chat.completions, "create", return_value=openai_response), \
             patch.object(lp.cohere_client, "chat", return_value=cohere_response):
            summarizer = NewsSummarizer()
            results = summarizer.process_articles(count=2)

        assert len(results) == 2
        for result in results:
            assert result["summary"] == "A concise summary."
            assert result["summary_provider"] == "openai"
            assert result["sentiment"] == "Neutral"
            assert result["sentiment_provider"] == "cohere"

    def test_process_articles_returns_empty_list_on_fetch_failure(self):
        with patch.object(NewsAPIClient, "get_top_headlines", side_effect=NewsAPIError("boom")):
            summarizer = NewsSummarizer()
            results = summarizer.process_articles(count=2)

        assert results == []

    def test_summarize_falls_back_to_cohere_when_openai_fails(self, cohere_response):
        with patch.object(NewsAPIClient, "get_top_headlines", return_value=[]), \
             patch.object(lp.openai_client.chat.completions, "create", side_effect=Exception("down")), \
             patch.object(lp.cohere_client, "chat", return_value=cohere_response):
            summarizer = NewsSummarizer()
            result = summarizer.summarize_article({"title": "T", "description": "D"})

        assert result["provider"] == "cohere"
        assert result["response"] == "Neutral"