from __future__ import annotations

import json
from typing import Any, Dict, List

from google import genai
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig

from research_assistant.utils.json_parsing import parse_jsonish


class SynthesisAgent(LlmAgent):
    def __init__(self, model: str):
        instruction = """You are a synthesis agent.

Combine the refined draft + fact-check notes into a coherent report-ready narrative.

Return STRICT JSON:
{
  "executive_summary": "1-3 sentences",
  "synthesis": "3-6 paragraphs",
  "key_insights": ["...","...","..."],
  "themes": ["...","..."],
  "coherence_score": 0.0-1.0
}
Return ONLY JSON."""

        super().__init__(
            name="synthesis",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.6,
                max_output_tokens=1600,
                response_mime_type="application/json",
            ),
        )

    def synthesize(
        self,
        *,
        client: genai.Client,
        query: str,
        refined: Dict[str, Any],
        fact_check: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = f"""{self.instruction}

user: Synthesize a report section.

QUESTION: {query}

REFINED_DRAFT_JSON:
{json.dumps(refined, indent=2)}

FACT_CHECK_JSON:
{json.dumps(fact_check, indent=2)}

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
                "executive_summary": "",
                "synthesis": refined.get("answer", ""),
                "key_insights": refined.get("key_points", []),
                "themes": [],
                "coherence_score": 0.5,
                "_metadata": {"agent": self.name, "error": "parse_error"},
            }

