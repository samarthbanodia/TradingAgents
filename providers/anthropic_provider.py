"""Anthropic LLM provider."""

from .base import LLMProvider
from anthropic import Anthropic


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: str = None,
                 temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__(model, api_key, temperature, max_tokens)
        self.client = Anthropic(api_key=api_key)

    def _raw_call(self, system_prompt: str, user_prompt: str) -> str:
        # Pre-fill assistant turn with "{" to force pure JSON output (no markdown preamble)
        resp = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "{"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return "{" + resp.content[0].text
