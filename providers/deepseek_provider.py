"""DeepSeek LLM provider (OpenAI-compatible API)."""

from .base import LLMProvider
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(LLMProvider):
    def __init__(self, model: str = "deepseek-chat", api_key: str = None,
                 temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__(model, api_key, temperature, max_tokens)
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def _raw_call(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content
