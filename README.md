# Multi-Provider News Summarizer

A small pipeline that fetches live news articles, summarizes each one with OpenAI, and runs sentiment analysis on the summary with Cohere, with fallback logic so a failure in one provider doesn't stop the pipeline, and cost tracking on every request.

## What it does

1. **Fetches articles** from [NewsAPI](https://newsapi.org) for a chosen category (technology, business, science, etc.).
2. **Summarizes each article** in 2-3 sentences using OpenAI (`gpt-4o-mini`). If OpenAI fails, the summary step automatically falls back to Cohere.
3. **Analyzes sentiment** of the summary (Positive / Neutral / Negative + reason) using Cohere (`command-r-08-2024`). If Cohere fails, this step falls back to OpenAI.
4. **Tracks cost** on every single request using the exact token counts returned by each provider's API (not estimates), and enforces a configurable daily budget.
5. **Respects rate limits**: the NewsAPI client retries with exponential backoff on `429` responses instead of failing immediately.

## Project structure

```
news-summarizer/
├── .env                  # your real API keys (never committed)
├── .env.example          # template with placeholder values
├── .gitignore
├── requirements.txt
├── config.py             # loads/validates .env settings
├── news_api.py           # NewsAPI client with retries
├── llm_providers.py      # OpenAI + Cohere calls, budget tracking, fallback
├── summarizer.py         # NewsSummarizer pipeline (fetch -> summarize -> sentiment)
├── test_summarizer.py    # unit tests (all external calls mocked)
└── main.py               # interactive CLI entry point
```

## Setup

1. Create and activate a virtual environment (conda or venv):
   ```
   conda create -n news-summarizer python=3.12
   conda activate news-summarizer
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in real values:
   ```
   cp .env.example .env
   ```
   You'll need:
   - `OPENAI_API_KEY`, from [platform.openai.com](https://platform.openai.com)
   - `COHERE_API_KEY`, from [dashboard.cohere.com](https://dashboard.cohere.com)
   - `NEWS_API_KEY`, from [newsapi.org](https://newsapi.org)
4. Validate your setup:
   ```
   python config.py
   ```
   This should print `Configuration valid.` If not, it tells you exactly which key is missing.

## How to run

Run each module standalone to check a specific piece:

```
python news_api.py         # fetch and print 3 tech articles
python llm_providers.py    # test both providers + show cost tracking
python summarizer.py       # fetch 2 articles, summarize + sentiment, cost summary
```

Run the full interactive app:

```
python main.py
```

You'll be prompted to pick a category and how many articles (1-10) to process, then it loops so you can run another batch without restarting.

Run the test suite:

```
pytest test_summarizer.py -v
```

## Example output

```
Fetching and processing 2 'technology' article(s)...

1. Tech giant unveils new AI chip promising major speed gains
   URL:               https://example.com/article-1
   Summary (openai):  A major tech company announced a new AI chip claimed to
                       significantly outperform previous generations. The
                       release is expected to accelerate machine learning
                       workloads across the industry.
   Sentiment (cohere): Positive. The tone is optimistic about the chip's impact.

2. Startup faces backlash after abrupt layoffs
   URL:               https://example.com/article-2
   Summary (openai):  A startup laid off a large portion of its workforce
                       with little notice, drawing criticism from employees
                       and industry observers.
   Sentiment (cohere): Negative. The coverage centers on employee backlash.

============================================================
TOKEN BUDGET SUMMARY
============================================================
Daily budget: $5.00
Used:         $0.000230 (0.00%)
Remaining:    $4.999770

  OPENAI
    Input tokens:  207
    Output tokens: 176
    Cost:          $0.000137

  COHERE
    Input tokens:  264
    Output tokens: 89
    Cost:          $0.000093
============================================================
```

## Cost analysis

Cost is tracked per-request using each provider's actual returned token counts, priced at:

| Provider | Model | Input ($/1M tokens) | Output ($/1M tokens) |
|---|---|---|---|
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Cohere | command-r-08-2024 | $0.15 | $0.60 |

For a typical article (short title + description in, a 2-3 sentence summary out), summarization runs roughly 200-400 input tokens and 60-120 output tokens, a fraction of a cent per article on OpenAI. Sentiment analysis is a shorter prompt (the summary itself) but priced about 15x higher per token on Cohere's `command-r`, so it often ends up costing more per call than the OpenAI summarization step despite using fewer tokens, as the example above shows (COHERE cost > OPENAI cost even with fewer tokens processed).

Processing a batch of 5 articles typically costs well under $0.01 total. The `DAILY_BUDGET` setting in `.env` (default `$5.00`) is a hard ceiling. If a request would exceed it, `TokenBudgetManager` raises an error instead of making the call, so a runaway loop can't rack up unexpected charges.
