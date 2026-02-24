"""Abstract base class for LLM providers with retry logic and JSON extraction."""

import abc
import json
import re
import time
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_CODES = (429, 500, 502, 503)


def extract_json(text: str) -> dict:
    """Try json.loads first, then extract from ```json blocks, then find first { ... }."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # First { ... } (greedy from first { to last })
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


class LLMProvider(abc.ABC):
    def __init__(self, model: str, api_key: str, temperature: float = 0.3, max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abc.abstractmethod
    def _raw_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make the actual API call. Returns raw text response."""
        ...

    def call(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the LLM with retry logic and JSON extraction."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = self._raw_call(system_prompt, user_prompt)
                parsed = extract_json(raw)
                if parsed is not None:
                    return parsed
                # JSON parse failed — retry with explicit instruction
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"[{self.model}] Invalid JSON on attempt {attempt+1}, retrying with explicit instruction")
                    raw = self._raw_call(
                        system_prompt,
                        user_prompt + "\n\nIMPORTANT: Return valid JSON only. No markdown, no explanation."
                    )
                    parsed = extract_json(raw)
                    if parsed is not None:
                        return parsed
                    raise ValueError(f"Invalid JSON after retry: {raw[:200]}")
                else:
                    raise ValueError(f"Invalid JSON after {MAX_RETRIES} attempts: {raw[:200]}")

            except Exception as e:
                last_error = e
                status = getattr(e, "status_code", getattr(e, "status", None))
                if status in RETRY_CODES and attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"[{self.model}] Retryable error (status={status}), waiting {wait}s: {e}")
                    time.sleep(wait)
                    continue
                # For non-retryable or last attempt, check if it's a JSON error we can still retry
                if "Invalid JSON" in str(e) and attempt < MAX_RETRIES - 1:
                    continue
                raise

        raise last_error
