from __future__ import annotations

import time
from typing import Any, Dict

from research_assistant.config.settings import Settings
from research_assistant.runtime.genai_client import create_client
from research_assistant.agents.router import QueryRouterAgent
from research_assistant.agents.sources import collect_sources
from research_assistant.agents.refinement import refine_answer
from research_assistant.agents.validation import FactCheckAgent
from research_assistant.agents.synthesis import SynthesisAgent
from research_assistant.agents.citations import CitationAgent
from research_assistant.evaluation.metrics import Metrics


async def run_research_pipeline(*, query: str, settings: Settings) -> Dict[str, Any]:
    """
    Execute the end-to-end workflow and return structured outputs.

    This is designed as a portfolio example: explicit stage outputs and clear data handoffs.
    """
    t0 = time.time()
    client = create_client(settings)

    router = QueryRouterAgent(model=settings.model_name)
    route = router.route(client=client, query=query)

    sources_bundle = await collect_sources(client=client, query=query, model=settings.model_name, settings=settings)
    top_sources = sources_bundle.get("top_sources", []) or []

    refinement = await refine_answer(
        client=client,
        settings=settings,
        query=query,
        sources=top_sources,
        model=settings.model_name,
    )

    fact_checker = FactCheckAgent(model=settings.model_name)
    fact_check = fact_checker.check(client=client, answer=refinement.get("final", {}) or {}, sources=top_sources)

    synthesizer = SynthesisAgent(model=settings.model_name)
    synthesis = synthesizer.synthesize(
        client=client,
        query=query,
        refined=refinement.get("final", {}) or {},
        fact_check=fact_check,
        sources=top_sources,
    )

    citer = CitationAgent(model=settings.model_name)
    citations = citer.format(client=client, sources=top_sources)

    wall = time.time() - t0
    metrics = Metrics()
    best_score = 0.0
    try:
        history = refinement.get("history") or []
        if history:
            best_score = max(float((h.get("critique") or {}).get("quality_score", 0.0) or 0.0) for h in history)
    except Exception:
        best_score = 0.0

    metrics.record(
        quality=best_score,
        sources=len(top_sources),
        iters=int(refinement.get("iterations", 0) or 0),
        credibility=float(fact_check.get("credibility_score", 0.0) or 0.0),
        citations=int(citations.get("total_citations", 0) or 0),
        wall_time_s=wall,
    )

    return {
        "query": query,
        "route": route,
        "sources": sources_bundle,
        "refinement": refinement,
        "fact_check": fact_check,
        "synthesis": synthesis,
        "citations": citations,
        "metrics": metrics.summary(),
        "runtime_s": wall,
    }

