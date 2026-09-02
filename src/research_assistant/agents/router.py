from __future__ import annotations

import json
from typing import Any, Dict

from google import genai
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig

from research_assistant.utils.json_parsing import parse_jsonish


class QueryRouterAgent(LlmAgent):
    """
    Lightweight router that classifies a query and recommends a strategy.

    Portfolio note: this is intentionally generic and can be extended to route to
    different workflows (e.g., legal, medical, software, etc.).
    """

    def __init__(self, model: str):
        instruction = """You are a query routing specialist.

Return STRICT JSON:
{
  "domain": "computer_science|life_science|physical_science|economics|general",
  "complexity": "low|medium|high",
  "confidence": 0.0-1.0,
  "recommended_sources": ["web_like","papers_like","scholar_like"],
  "notes": "one short sentence"
}

Return ONLY JSON."""

        super().__init__(
            name="query_router",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=384,
                response_mime_type="application/json",
            ),
        )

    def route(self, *, client: genai.Client, query: str) -> Dict[str, Any]:
        prompt = f"{self.instruction}\n\nuser: Route this query: {query}"
        resp = client.models.generate_content(model=self.model, contents=prompt, config=self.generate_content_config)
        try:
            parsed = parse_jsonish(resp.text)
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError("non-dict", resp.text or "", 0)
            parsed["_metadata"] = {"agent": self.name}
            return parsed
        except Exception:
            return {
                "domain": "general",
                "complexity": "medium",
                "confidence": 0.5,
                "recommended_sources": ["web_like", "papers_like", "scholar_like"],
                "notes": "Fallback route (unable to parse router output).",
                "_metadata": {"agent": self.name, "error": "parse_error"},
            }

