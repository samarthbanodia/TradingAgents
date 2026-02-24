"""Google Gemini LLM provider."""

from .base import LLMProvider
from google import genai


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None,
                 temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__(model, api_key, temperature, max_tokens)
        self.client = genai.Client(api_key=api_key)

    def _raw_call(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
        except TypeError:
            # Fallback for older SDK versions that don't support system_instruction
            resp = self.client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\n---\n\n{user_prompt}",
                config=genai.types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
        return resp.text
