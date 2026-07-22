"""
config.py

Central configuration module for the multi-provider news summarizer.

Loads settings from a local .env file (via python-dotenv) and exposes them
as simple module-level constants, plus a validate_config() function that
checks required API keys are present before the rest of the app runs.

Run directly (`python config.py`) to check your .env setup:
    - Prints "Configuration valid" if everything required is present.
    - Otherwise lists exactly which required keys are missing.
"""

import os
import sys

from dotenv import load_dotenv

# Load variables from .env into the process environment.
# override=False means real environment variables (e.g. set in CI) win
# over whatever is in the .env file.
load_dotenv(override=False)


# ---------------------------------------------------------------------------
# API keys (required — the app cannot function without these)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ---------------------------------------------------------------------------
# General settings (optional — sensible defaults if not set in .env)
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
DAILY_BUDGET = float(os.getenv("DAILY_BUDGET", 5.00))

# Keys that MUST be set for the app to run at all.
REQUIRED_KEYS = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "COHERE_API_KEY": COHERE_API_KEY,
    "NEWS_API_KEY": NEWS_API_KEY,
}

# Placeholder values from the .env.example / instructions — if these were
# left in place instead of being replaced with a real key, treat them as
# missing. Covers both styles used in this project's .env.example:
#   OPENAI_API_KEY=<your key from platform.openai.com>
#   NEWS_API_KEY=<NEWS_API_KEY>          (bracketed key name)
#   NEWS_API_KEY=NEWS_API_KEY            (bare key name, no brackets)
PLACEHOLDER_MARKERS = ("<your key", "your_", "changeme", "xxx")


def _looks_like_placeholder(value: str, key_name: str = None) -> bool:
    """Return True if a config value looks like an unfilled placeholder."""
    if not value:
        return True

    stripped = value.strip()
    lowered = stripped.lower()

    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        return True

    # Catches "<ANYTHING>" style placeholders, e.g. <NEWS_API_KEY>
    if stripped.startswith("<") and stripped.endswith(">"):
        return True

    # Catches a value that's just the key's own name, e.g.
    # NEWS_API_KEY=NEWS_API_KEY
    if key_name and lowered == key_name.lower():
        return True

    return False


def validate_config():
    """
    Validate that all required configuration values are present and look
    like real keys (not leftover placeholders from .env.example).

    Returns:
        (is_valid: bool, missing: list[str]) — is_valid is True only if
        every required key is set and non-placeholder. missing lists the
        names of any keys that failed validation.
    """
    missing = [
        name
        for name, value in REQUIRED_KEYS.items()
        if _looks_like_placeholder(value, key_name=name)
    ]
    return (len(missing) == 0, missing)


def print_config_summary():
    """Print a non-sensitive summary of the loaded configuration."""
    print(f"Environment:      {ENVIRONMENT}")
    print(f"Max retries:      {MAX_RETRIES}")
    print(f"Request timeout:  {REQUEST_TIMEOUT}s")
    print(f"Daily budget:     ${DAILY_BUDGET:.2f}")
    for name in REQUIRED_KEYS:
        is_placeholder = _looks_like_placeholder(REQUIRED_KEYS[name], key_name=name)
        print(f"{name:<20} {'MISSING' if is_placeholder else 'set'}")


if __name__ == "__main__":
    is_valid, missing_keys = validate_config()
    print_config_summary()
    print()

    if is_valid:
        print("Configuration valid.")
        sys.exit(0)
    else:
        print("Missing required configuration:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nCheck your .env file and make sure each key above is set to a real value.")
        sys.exit(1)