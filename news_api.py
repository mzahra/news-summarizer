"""
news_api.py

Thin client for NewsAPI.org (https://newsapi.org).

Fetches top headlines for a given category/country, with basic retry
handling for transient failures and rate limits.

Run directly (`python news_api.py`) to fetch and print 3 tech articles.
"""

import time

import requests

import config


class NewsAPIError(Exception):
    """Raised when the News API request fails after all retries."""


class NewsAPIClient:
    """Minimal client for fetching top headlines from NewsAPI.org."""

    BASE_URL = "https://newsapi.org/v2/top-headlines"

    def __init__(self, api_key=None, timeout=None, max_retries=None):
        self.api_key = api_key or config.NEWS_API_KEY
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self.max_retries = max_retries or config.MAX_RETRIES

        if not self.api_key:
            raise NewsAPIError(
                "NEWS_API_KEY is not set. Add it to your .env file."
            )

    def get_top_headlines(self, category="technology", country="us", page_size=3):
        """
        Fetch top headlines for a category/country.

        Args:
            category: NewsAPI category (e.g. "technology", "business").
            country: Two-letter country code (e.g. "us").
            page_size: Number of articles to return (max 100 per NewsAPI).

        Returns:
            A list of article dicts as returned by NewsAPI, each with
            keys like "title", "description", "url", "source", "publishedAt".

        Raises:
            NewsAPIError: if the request fails after all retries, or the
            API responds with an error status.
        """
        params = {
            "category": category,
            "country": country,
            "pageSize": page_size,
            "apiKey": self.api_key,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.BASE_URL, params=params, timeout=self.timeout
                )

                if response.status_code == 429:
                    # Rate limited — back off and retry.
                    wait = 2 ** attempt
                    last_error = NewsAPIError(
                        f"Rate limited by NewsAPI (attempt {attempt}/{self.max_retries})."
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                if data.get("status") != "ok":
                    raise NewsAPIError(
                        f"NewsAPI error: {data.get('code')} - {data.get('message')}"
                    )

                return data.get("articles", [])

            except requests.exceptions.RequestException as exc:
                last_error = NewsAPIError(f"Request failed (attempt {attempt}/{self.max_retries}): {exc}")
                time.sleep(min(2 ** attempt, 10))

        raise last_error or NewsAPIError("Failed to fetch news for an unknown reason.")


def _print_articles(articles):
    """Pretty-print a list of article dicts to the console."""
    for i, article in enumerate(articles, start=1):
        source = article.get("source", {}).get("name", "Unknown source")
        print(f"{i}. {article.get('title')}")
        print(f"   Source:    {source}")
        print(f"   Published: {article.get('publishedAt')}")
        print(f"   URL:       {article.get('url')}")
        print()


if __name__ == "__main__":
    is_valid, missing = config.validate_config()
    if not is_valid:
        print("Cannot run news_api.py — missing configuration:")
        for key in missing:
            print(f"  - {key}")
        raise SystemExit(1)

    client = NewsAPIClient()

    print("Fetching 3 tech news articles...\n")
    try:
        articles = client.get_top_headlines(category="technology", page_size=3)
    except NewsAPIError as e:
        print(f"Failed to fetch news: {e}")
        raise SystemExit(1)

    if not articles:
        print("No articles returned. Try a different category or country.")
    else:
        _print_articles(articles)