"""
llm_providers.py

Multi-provider LLM integration for the news summarizer:
    - OpenAI: article summarization
    - Cohere: sentiment analysis

Adapted from the course's integration-patterns notebook (which paired
OpenAI with Anthropic) — this lab swaps in Cohere per our .env setup.
Keeps the same core patterns: direct API usage tracking (exact token
counts from response objects, not estimates), a shared token/cost
budget manager, and a provider fallback helper.

Run directly (`python llm_providers.py`) to test both providers and
print a cost summary.
"""

import time
from typing import Any, Dict, Optional

from openai import OpenAI
import cohere

import config

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
cohere_client = cohere.ClientV2(api_key=config.COHERE_API_KEY) if cohere else None

OPENAI_MODEL = "gpt-4o-mini"
COHERE_MODEL = "command-r-08-2024"

# Pricing per token, derived from published per-million-token rates.
# Update here if provider pricing changes.
PRICING = {
    "openai": {
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    },
    "cohere": {
        "command-r-08-2024": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "command-r-plus-08-2024": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    },
}


class BudgetExceededError(Exception):
    """Raised when a request would push spend past the configured daily budget."""


class ProviderError(Exception):
    """Raised when a provider call fails after all retries."""


# ---------------------------------------------------------------------------
# Token / cost tracking
# ---------------------------------------------------------------------------
class TokenBudgetManager:
    """Tracks token usage and cost across providers using exact API usage data."""

    def __init__(self, daily_budget: Optional[float] = None):
        self.daily_budget = daily_budget if daily_budget is not None else config.DAILY_BUDGET
        self.used_budget = 0.0
        self.provider_usage: Dict[str, Dict[str, float]] = {
            "openai": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            "cohere": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        }

    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in dollars for a request using exact token counts."""
        rates = PRICING.get(provider, {}).get(model)
        if not rates:
            return 0.0
        return input_tokens * rates["input"] + output_tokens * rates["output"]

    def track_request(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """
        Record a request's cost against the daily budget.

        Raises:
            BudgetExceededError: if this request would push spend over the
            daily budget.
        """
        cost = self.calculate_cost(provider, model, input_tokens, output_tokens)

        if self.used_budget + cost > self.daily_budget:
            remaining = self.daily_budget - self.used_budget
            raise BudgetExceededError(
                f"Daily budget exceeded: this request costs ${cost:.6f}, "
                f"only ${remaining:.6f} remaining of ${self.daily_budget:.2f}."
            )

        self.used_budget += cost
        usage = self.provider_usage.setdefault(
            provider, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        )
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["cost"] += cost

        return {
            "cost": cost,
            "remaining_budget": self.daily_budget - self.used_budget,
            "budget_used_percent": (self.used_budget / self.daily_budget * 100) if self.daily_budget else 0,
        }

    def print_summary(self):
        """Print a cost summary across all tracked providers."""
        pct = (self.used_budget / self.daily_budget * 100) if self.daily_budget else 0
        print("\n" + "=" * 60)
        print("TOKEN BUDGET SUMMARY")
        print("=" * 60)
        print(f"Daily budget: ${self.daily_budget:.2f}")
        print(f"Used:         ${self.used_budget:.6f} ({pct:.2f}%)")
        print(f"Remaining:    ${self.daily_budget - self.used_budget:.6f}")
        for provider, usage in self.provider_usage.items():
            if usage["input_tokens"] or usage["output_tokens"]:
                print(f"\n  {provider.upper()}")
                print(f"    Input tokens:  {int(usage['input_tokens']):,}")
                print(f"    Output tokens: {int(usage['output_tokens']):,}")
                print(f"    Cost:          ${usage['cost']:.6f}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Raw provider calls — each returns {"response": str, "usage": {...}}
# using exact token counts straight from the API response (no estimation).
# ---------------------------------------------------------------------------
def ask_openai(prompt: str, model: str = OPENAI_MODEL) -> Dict[str, Any]:
    """Send a prompt to OpenAI and return the response with exact token usage."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=config.REQUEST_TIMEOUT,
    )
    usage = response.usage
    return {
        "response": response.choices[0].message.content,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


def ask_cohere(prompt: str, model: str = COHERE_MODEL) -> Dict[str, Any]:
    """Send a prompt to Cohere and return the response with exact (billed) token usage."""
    if cohere_client is None:
        raise ImportError(
            "The 'cohere' package is not installed. Install dependencies with "
            "'pip install -r requirements.txt'."
        )

    response = cohere_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(
        block.text for block in response.message.content if getattr(block, "type", None) == "text"
    )

    billed = response.usage.billed_units
    input_tokens = int(billed.input_tokens or 0)
    output_tokens = int(billed.output_tokens or 0)

    return {
        "response": text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Retry wrapper — used by the budget-aware helpers below
# ---------------------------------------------------------------------------
def _call_with_retries(func, *args, max_retries: Optional[int] = None, **kwargs):
    """Call func with simple exponential backoff retries."""
    retries = max_retries if max_retries is not None else config.MAX_RETRIES
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))

    raise ProviderError(f"{func.__name__} failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Budget-aware helpers — call a provider, track cost, print a running total
# ---------------------------------------------------------------------------
def ask_openai_with_budget(
    prompt: str, model: str = OPENAI_MODEL, budget_manager: Optional[TokenBudgetManager] = None
) -> Dict[str, Any]:
    """Ask OpenAI and track exact token cost against a TokenBudgetManager."""
    result = _call_with_retries(ask_openai, prompt, model)

    if budget_manager:
        budget_info = budget_manager.track_request(
            provider="openai",
            model=model,
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
        )
        print(
            f"[openai] {result['usage']['input_tokens']} in + "
            f"{result['usage']['output_tokens']} out tokens "
            f"= ${budget_info['cost']:.6f} (${budget_info['remaining_budget']:.4f} remaining)"
        )

    return {**result, "provider": "openai", "model": model}


def ask_cohere_with_budget(
    prompt: str, model: str = COHERE_MODEL, budget_manager: Optional[TokenBudgetManager] = None
) -> Dict[str, Any]:
    """Ask Cohere and track exact token cost against a TokenBudgetManager."""
    result = _call_with_retries(ask_cohere, prompt, model)

    if budget_manager:
        budget_info = budget_manager.track_request(
            provider="cohere",
            model=model,
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
        )
        print(
            f"[cohere] {result['usage']['input_tokens']} in + "
            f"{result['usage']['output_tokens']} out tokens "
            f"= ${budget_info['cost']:.6f} (${budget_info['remaining_budget']:.4f} remaining)"
        )

    return {**result, "provider": "cohere", "model": model}


# ---------------------------------------------------------------------------
# Fallback pattern — try primary provider, fall back to secondary on failure
# ---------------------------------------------------------------------------
def ask_with_fallback(
    prompt: str,
    primary: str = "openai",
    secondary: str = "cohere",
    budget_manager: Optional[TokenBudgetManager] = None,
) -> Dict[str, Any]:
    """
    Try the primary provider first; fall back to the secondary provider
    if the primary raises an exception (rate limit, timeout, etc.).
    """
    providers = {
        "openai": ask_openai_with_budget,
        "cohere": ask_cohere_with_budget,
    }

    try:
        print(f"Trying {primary} (primary)...")
        result = providers[primary](prompt, budget_manager=budget_manager)
        print(f"{primary} succeeded")
        return result
    except Exception as exc:
        print(f"{primary} failed: {exc}")
        print(f"Falling back to {secondary}...")

    result = providers[secondary](prompt, budget_manager=budget_manager)
    print(f"{secondary} succeeded")
    return result


if __name__ == "__main__":
    is_valid, missing = config.validate_config()
    if not is_valid:
        print("Cannot run llm_providers.py — missing configuration:")
        for key in missing:
            print(f"  - {key}")
        raise SystemExit(1)

    budget_manager = TokenBudgetManager()
    cohere_available = cohere_client is not None

    print("Testing OpenAI (summarization role)...")
    openai_result = ask_openai_with_budget(
        "Summarize in one sentence: OpenAI and Cohere can be combined in a pipeline "
        "where one model drafts text and another critiques it.",
        budget_manager=budget_manager,
    )
    print(f"Response: {openai_result['response']}\n")

    if cohere_available:
        print("Testing Cohere (sentiment analysis role)...")
        cohere_result = ask_cohere_with_budget(
            "What is the sentiment of this sentence, in one word: "
            "'The new update made everything so much faster and easier to use.'",
            budget_manager=budget_manager,
        )
        print(f"Response: {cohere_result['response']}\n")
    else:
        print("Skipping Cohere test: the 'cohere' package is not installed.\n")

    budget_manager.print_summary()
