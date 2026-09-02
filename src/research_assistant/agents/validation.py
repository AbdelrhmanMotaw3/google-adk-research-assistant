from __future__ import annotations

import json
from typing import Any, Dict, List

from google import genai
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig

from research_assistant.utils.json_parsing import parse_jsonish


class FactCheckAgent(LlmAgent):
    def __init__(self, model: str):
        instruction = """You are a fact-checking assistant.

Given a draft answer and a list of sources, flag claims that seem unsupported.

Return STRICT JSON:
{
  "credibility_score": 0.0-1.0,
  "verified_claims": ["..."],
  "questionable_claims": ["..."],
  "notes": "1-3 sentences"
}
Return ONLY JSON."""

        super().__init__(
            name="fact_check",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=900,
                response_mime_type="application/json",
            ),
        )

    def check(self, *, client: genai.Client, answer: Dict[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""{self.instruction}

user: Fact-check this answer:

ANSWER_JSON:
{json.dumps(answer, indent=2)}

SOURCES_JSON:
{json.dumps(sources[:8], indent=2) if sources else "[]"}
"""
        resp = client.models.generate_content(model=self.model, contents=prompt, config=self.generate_content_config)
        try:
            parsed = parse_jsonish(resp.text)
            if not isinstance(parsed, dict):
                raise ValueError("non-dict")
            parsed["_metadata"] = {"agent": self.name}
            return parsed
        except Exception:
            return {
                "credibility_score": 0.5,
                "verified_claims": [],
                "questionable_claims": [],
                "notes": "Fallback fact-check (unable to parse output).",
                "_metadata": {"agent": self.name, "error": "parse_error"},
            }

