from __future__ import annotations

import json
from typing import Any, Dict, List

from google import genai
from google.adk.agents import LlmAgent, ParallelAgent
from google.genai.types import GenerateContentConfig

from research_assistant.runtime.adk_runner import last_final_text, run_agent_events
from research_assistant.config.settings import Settings
from research_assistant.utils.json_parsing import parse_jsonish


def _normalize_search_payload(source_type: str, payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"source_type": source_type, "results": payload, "total_found": len(payload), "search_time": 0.0}
    if isinstance(payload, dict):
        results = payload.get("results", [])
        if not isinstance(results, list):
            results = []
        return {
            "source_type": source_type,
            "results": results,
            "total_found": len(results),
            "search_time": float(payload.get("search_time", 0.0) or 0.0),
        }
    return {"source_type": source_type, "results": [], "total_found": 0, "search_time": 0.0}


def _derive_top_sources(search_results: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    derived: List[Dict[str, Any]] = []
    seen = set()
    for r in search_results:
        st = (r.get("source_type") or "unknown").lower()
        for item in (r.get("results") or []):
            if not isinstance(item, dict):
                continue
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            if not url or not title or url in seen:
                continue
            seen.add(url)
            derived.append(
                {
                    "title": title,
                    "type": st,
                    "url": url,
                    "relevance_score": float(item.get("relevance_score", item.get("relevance", 0.0)) or 0.0),
                    "snippet": (item.get("snippet") or item.get("abstract") or "").strip(),
                }
            )
    derived.sort(key=lambda x: float(x.get("relevance_score", 0.0) or 0.0), reverse=True)
    return derived[:limit]


class WebLikeSourceAgent(LlmAgent):
    def __init__(self, model: str):
        instruction = """You are a web-like source discovery agent.

Generate 3-5 plausible results with title, url, snippet, relevance (0-1).
Return STRICT JSON:
{
  "source_type": "web_like",
  "results": [{"title":"...","url":"https://...","snippet":"...","relevance":0.9}],
  "search_time": 0.0
}
Return ONLY JSON."""

        super().__init__(
            name="web_like_sources",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )

    def search(self, client: genai.Client, query: str) -> Dict[str, Any]:
        resp = client.models.generate_content(
            model=self.model,
            contents=f"{self.instruction}\n\nuser: Query: {query}",
            config=self.generate_content_config,
        )
        return _normalize_search_payload("web_like", parse_jsonish(resp.text))


class PapersLikeSourceAgent(LlmAgent):
    def __init__(self, model: str):
        instruction = """You are a papers-like source discovery agent.

Generate 3-5 plausible academic items with title, authors, url, abstract, relevance (0-1).
Return STRICT JSON:
{
  "source_type": "papers_like",
  "results": [{"title":"...","authors":["A","B"],"url":"https://...","abstract":"...","relevance":0.9}],
  "search_time": 0.0
}
Return ONLY JSON."""

        super().__init__(
            name="papers_like_sources",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1200,
                response_mime_type="application/json",
            ),
        )

    def search(self, client: genai.Client, query: str) -> Dict[str, Any]:
        resp = client.models.generate_content(
            model=self.model,
            contents=f"{self.instruction}\n\nuser: Query: {query}",
            config=self.generate_content_config,
        )
        return _normalize_search_payload("papers_like", parse_jsonish(resp.text))


class ScholarLikeSourceAgent(LlmAgent):
    def __init__(self, model: str):
        instruction = """You are a scholar-like source discovery agent.

Generate 3-5 plausible items with title, venue, year, url, snippet, citations, relevance (0-1).
Return STRICT JSON:
{
  "source_type": "scholar_like",
  "results": [{"title":"...","venue":"...","year":2024,"url":"https://...","snippet":"...","citations":10,"relevance":0.9}],
  "search_time": 0.0
}
Return ONLY JSON."""

        super().__init__(
            name="scholar_like_sources",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1200,
                response_mime_type="application/json",
            ),
        )

    def search(self, client: genai.Client, query: str) -> Dict[str, Any]:
        resp = client.models.generate_content(
            model=self.model,
            contents=f"{self.instruction}\n\nuser: Query: {query}",
            config=self.generate_content_config,
        )
        return _normalize_search_payload("scholar_like", parse_jsonish(resp.text))


async def collect_sources(*, client: genai.Client, query: str, model: str, settings: Settings) -> Dict[str, Any]:
    """
    Fan-out/fan-in source discovery.

    Returns a bundle with:
    - raw_searches: per-worker outputs
    - top_sources: merged list used by later stages
    """
    web = WebLikeSourceAgent(model=model)
    papers = PapersLikeSourceAgent(model=model)
    scholar = ScholarLikeSourceAgent(model=model)

    parallel = ParallelAgent(name="source_fanout", sub_agents=[web, papers, scholar])

    events = await run_agent_events(parallel, message=query, settings=settings, app_name="sources")

    name_to_type = {web.name: "web_like", papers.name: "papers_like", scholar.name: "scholar_like"}
    search_results: List[Dict[str, Any]] = []
    for agent in parallel.sub_agents:
        raw = last_final_text(events, author=agent.name)
        try:
            parsed = parse_jsonish(raw)
        except Exception:
            # fallback to direct execution path
            parsed = agent.search(client, query) if hasattr(agent, "search") else {}
        st = name_to_type.get(agent.name, agent.name)
        search_results.append(_normalize_search_payload(st, parsed))

    top_sources = _derive_top_sources(search_results, limit=12)

    return {
        "raw_searches": search_results,
        "top_sources": top_sources,
        "counts": {r["source_type"]: int(r.get("total_found", 0) or 0) for r in search_results},
    }

