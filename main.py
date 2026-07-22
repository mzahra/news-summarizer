"""
main.py

Interactive entry point for the multi-provider news summarizer.

Lets you pick a news category and how many articles to process, then
runs each through the OpenAI -> Cohere summarize + sentiment pipeline
and shows a running cost summary. Loops so you can process another
batch without restarting.

Run: python main.py
"""

import sys

import config
from llm_providers import TokenBudgetManager
from summarizer import NewsSummarizer

CATEGORIES = [
    "technology",
    "business",
    "general",
    "science",
    "health",
    "sports",
    "entertainment",
]


def prompt_category() -> str:
    """Ask the user to pick a NewsAPI category, defaulting to 'technology'."""
    print("\nAvailable categories:")
    for i, cat in enumerate(CATEGORIES, start=1):
        print(f"  {i}. {cat}")

    choice = input(f"Choose a category [1-{len(CATEGORIES)}] (default: technology): ").strip()
    if not choice:
        return "technology"

    try:
        index = int(choice) - 1
        if 0 <= index < len(CATEGORIES):
            return CATEGORIES[index]
    except ValueError:
        pass

    print("Invalid choice, defaulting to 'technology'.")
    return "technology"


def prompt_article_count() -> int:
    """Ask how many articles to process (1-10), defaulting to 3."""
    choice = input("How many articles to process? (1-10, default: 3): ").strip()
    if not choice:
        return 3

    try:
        count = int(choice)
        if 1 <= count <= 10:
            return count
    except ValueError:
        pass

    print("Invalid number, defaulting to 3.")
    return 3


def run_session(summarizer: NewsSummarizer):
    """Run one fetch-summarize-analyze cycle based on user input."""
    category = prompt_category()
    count = prompt_article_count()

    print(f"\nFetching and processing {count} '{category}' article(s)...\n")
    results = summarizer.process_articles(count=count, category=category)

    if not results:
        print("No articles were processed. Check your NEWS_API_KEY and category.")
        return

    summarizer.print_results(results)
    summarizer.budget_manager.print_summary()


def main():
    print("=" * 60)
    print("Multi-Provider News Summarizer")
    print("=" * 60)

    is_valid, missing = config.validate_config()
    if not is_valid:
        print("\nMissing required configuration:")
        for key in missing:
            print(f"  - {key}")
        print("\nCheck your .env file and try again.")
        sys.exit(1)

    budget_manager = TokenBudgetManager()
    summarizer = NewsSummarizer(budget_manager=budget_manager)

    while True:
        try:
            run_session(summarizer)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as exc:
            print(f"\nSomething went wrong: {exc}")

        again = input("\nProcess another batch? [y/N]: ").strip().lower()
        if again != "y":
            break

    print("\nSession complete. Final cost summary:")
    budget_manager.print_summary()


if __name__ == "__main__":
    main()