from __future__ import annotations

import json
from typing import Any, Dict, List

from google import genai
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig

from research_assistant.utils.json_parsing import parse_jsonish


def _looks_placeholder(bibliography: str) -> bool:
    b = (bibliography or "").strip().lower()
    if not b:
        return True
    placeholders = ["lorem ipsum", "placeholder", "n/a", "smith, j.", "doe, a."]
    return any(p in b for p in placeholders)


def _derive_from_sources(top_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    citations: List[Dict[str, Any]] = []
    bib: List[str] = []
    for idx, src in enumerate(top_sources, 1):
        title = (src.get("title") or "").strip() or "Untitled source"
        url = (src.get("url") or "").strip()
        author = (src.get("source") or src.get("publisher") or "Unknown author").strip()
        apa = f"{author} (n.d.). {title}. {url}".strip()
        citations.append(
            {
                "source_title": title,
                "citation_apa": apa,
                "citation_number": idx,
            }
        )
        bib.append(apa)
    return {
        "citations": citations,
        "bibliography": "\n".join(bib).strip(),
        "total_citations": len(citations),
        "citation_style": "APA",
        "_metadata": {"derived": True},
    }


class CitationAgent(LlmAgent):
    def __init__(self, model: str):
        instruction = """You are a citation formatter.

Given a list of sources, produce APA citations.

Return STRICT JSON:
{
  "citations": [{"source_title":"...","citation_apa":"...","citation_number":1}],
  "bibliography": "APA bibliography",
  "total_citations": 0,
  "citation_style": "APA"
}
Return ONLY JSON."""

        super().__init__(
            name="citations",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1200,
                response_mime_type="application/json",
            ),
        )

    def format(self, *, client: genai.Client, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""{self.instruction}

user: Format citations for:
{json.dumps(sources, indent=2)}
"""
        resp = client.models.generate_content(model=self.model, contents=prompt, config=self.generate_content_config)
        try:
            parsed = parse_jsonish(resp.text)
            if not isinstance(parsed, dict):
                raise ValueError("non-dict")
            bib = parsed.get("bibliography", "")
            citations = parsed.get("citations", [])
            if not isinstance(citations, list):
                citations = []

            # If the model produced low-quality output, fall back to deterministic derivation.
            if _looks_placeholder(str(bib)) or not citations:
                return _derive_from_sources(sources)

            parsed["total_citations"] = int(parsed.get("total_citations") or len(citations))
            parsed.setdefault("citation_style", "APA")
            parsed["_metadata"] = {"agent": self.name}
            return parsed
        except Exception:
            return _derive_from_sources(sources)

