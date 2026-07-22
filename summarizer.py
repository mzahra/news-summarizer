"""
summarizer.py

Core pipeline: fetch news articles, summarize each with OpenAI, then run
sentiment analysis on the summary with Cohere. Both steps use the
fallback pattern from llm_providers.py, so a failure in one provider
doesn't stop the pipeline — it just falls back to the other provider
for that step. All requests are tracked against a shared token/cost
budget.

Run directly (`python summarizer.py`) to process 2 articles end to end
and print a cost summary.
"""

from typing import Any, Dict, List, Optional

from llm_providers import TokenBudgetManager, ask_with_fallback
from news_api import NewsAPIClient, NewsAPIError

import config


class NewsSummarizer:
    """Fetches news articles and runs them through the summarize + sentiment pipeline."""

    def __init__(self, budget_manager: Optional[TokenBudgetManager] = None):
        self.news_client = NewsAPIClient()
        self.budget_manager = budget_manager or TokenBudgetManager()

    def summarize_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize a single article in 2-3 sentences.

        Primary: OpenAI. Falls back to Cohere if OpenAI fails.
        """
        title = article.get("title", "")
        description = article.get("description") or article.get("content") or ""

        prompt = (
            "Summarize this news article in 2-3 concise sentences.\n\n"
            f"Title: {title}\n\n"
            f"Details: {description}"
        )

        return ask_with_fallback(
            prompt,
            primary="openai",
            secondary="cohere",
            budget_manager=self.budget_manager,
        )

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Classify the sentiment of a piece of text as positive, neutral, or negative.

        Primary: Cohere. Falls back to OpenAI if Cohere fails.
        """
        prompt = (
            "Classify the sentiment of the following text as Positive, Neutral, "
            "or Negative, and give a one-sentence reason.\n\n"
            f"Text: {text}"
        )

        return ask_with_fallback(
            prompt,
            primary="cohere",
            secondary="openai",
            budget_manager=self.budget_manager,
        )

    def process_articles(self, count: int = 2, category: str = "technology") -> List[Dict[str, Any]]:
        """
        Fetch `count` articles and run each through summarize + sentiment analysis.

        Returns:
            A list of result dicts, one per article, each containing the
            original title/url plus the summary, sentiment, and which
            provider actually served each step (useful for spotting when
            fallback logic kicked in).
        """
        try:
            articles = self.news_client.get_top_headlines(category=category, page_size=count)
        except NewsAPIError as exc:
            print(f"Failed to fetch articles: {exc}")
            return []

        results = []
        for article in articles:
            summary_result = self.summarize_article(article)
            sentiment_result = self.analyze_sentiment(summary_result["response"])

            results.append(
                {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "summary": summary_result["response"],
                    "summary_provider": summary_result["provider"],
                    "sentiment": sentiment_result["response"],
                    "sentiment_provider": sentiment_result["provider"],
                }
            )

        return results

    def print_results(self, results: List[Dict[str, Any]]):
        """Pretty-print processed article results."""
        for i, r in enumerate(results, start=1):
            print(f"\n{i}. {r['title']}")
            print(f"   URL:               {r['url']}")
            print(f"   Summary ({r['summary_provider']}):  {r['summary']}")
            print(f"   Sentiment ({r['sentiment_provider']}): {r['sentiment']}")


if __name__ == "__main__":
    config.validate_or_exit("summarizer.py")

    summarizer = NewsSummarizer()

    print("Fetching and processing 2 articles...")
    results = summarizer.process_articles(count=2, category="technology")

    if not results:
        print("No articles were processed.")
    else:
        summarizer.print_results(results)
        summarizer.budget_manager.print_summary()
