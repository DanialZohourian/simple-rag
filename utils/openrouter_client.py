# utils/openrouter_client.py
import requests
from dataclasses import dataclass
from typing import Any

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

class OpenRouterHTTPError(RuntimeError):
    pass

@dataclass
class OpenRouterClient:
    api_key: str
    http_referer: str = ""
    x_title: str = ""

    def _headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            h["HTTP-Referer"] = self.http_referer
        if self.x_title:
            h["X-Title"] = self.x_title
        return h

    def embeddings(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        url = f"{OPENROUTER_BASE}/embeddings"
        payload = {"model": model, "input": inputs}
        r = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        if r.status_code >= 400:
            raise OpenRouterHTTPError(f"{r.status_code} {r.reason}: {r.text}")
        data = r.json()
        return [item["embedding"] for item in data["data"]]

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        url = f"{OPENROUTER_BASE}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=self._headers(), json=payload, timeout=180)
        if r.status_code >= 400:
            raise OpenRouterHTTPError(f"{r.status_code} {r.reason}: {r.text}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
