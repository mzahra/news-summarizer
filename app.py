"""
app.py

Simple Flask web UI for the multi-provider news summarizer.

Wraps the existing NewsSummarizer pipeline (fetch -> OpenAI summary ->
Cohere sentiment -> cost tracking) in a one-page form: pick a category
and article count, submit, see results and a running cost summary.

Run: python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

from flask import Flask, render_template, request

import config
from llm_providers import BudgetExceededError, TokenBudgetManager
from news_api import NewsAPIError
from summarizer import NewsSummarizer

config.validate_or_exit("app.py")

app = Flask(__name__)

CATEGORIES = [
    "technology",
    "business",
    "general",
    "science",
    "health",
    "sports",
    "entertainment",
]

# A single shared budget manager/summarizer for the life of the dev server,
# so the cost summary accumulates across page submissions the same way
# main.py accumulates it across loop iterations in one session. This is
# fine for a single-process local demo; a real multi-user deployment would
# need a per-session or per-user budget manager instead of one global one.
budget_manager = TokenBudgetManager()
summarizer = NewsSummarizer(budget_manager=budget_manager)


def _budget_context():
    """Build a plain-dict snapshot of the budget manager for the template."""
    return {
        "daily_budget": budget_manager.daily_budget,
        "used_budget": budget_manager.used_budget,
        "remaining_budget": budget_manager.daily_budget - budget_manager.used_budget,
        "provider_usage": budget_manager.provider_usage,
    }


@app.route("/", methods=["GET"])
def index():
    """Show the empty form (no results yet)."""
    return render_template(
        "index.html",
        categories=CATEGORIES,
        selected_category="technology",
        selected_count=3,
        results=None,
        error=None,
        budget=_budget_context(),
    )


@app.route("/process", methods=["POST"])
def process():
    """Fetch + summarize + score sentiment for the submitted category/count."""
    category = request.form.get("category", "technology")
    if category not in CATEGORIES:
        category = "technology"

    try:
        count = int(request.form.get("count", 3))
    except ValueError:
        count = 3
    count = max(1, min(count, 10))

    error = None
    results = []

    try:
        results = summarizer.process_articles(count=count, category=category)
        if not results:
            error = "No articles were processed. Check your NEWS_API_KEY and category."
    except BudgetExceededError as exc:
        error = f"Daily budget exceeded: {exc}"
    except NewsAPIError as exc:
        error = f"Failed to fetch articles: {exc}"
    except Exception as exc:
        error = f"Something went wrong: {exc}"

    return render_template(
        "index.html",
        categories=CATEGORIES,
        selected_category=category,
        selected_count=count,
        results=results,
        error=error,
        budget=_budget_context(),
    )


if __name__ == "__main__":
    app.run(debug=True)
