from __future__ import annotations

from typing import Any, Dict


def render_markdown_report(results: Dict[str, Any]) -> str:
    query = results.get("query", "")
    route = results.get("route", {}) or {}
    sources = results.get("sources", {}) or {}
    refinement = results.get("refinement", {}) or {}
    fact_check = results.get("fact_check", {}) or {}
    synthesis = results.get("synthesis", {}) or {}
    citations = results.get("citations", {}) or {}
    metrics = results.get("metrics", {}) or {}

    top_sources = sources.get("top_sources", []) or []
    refined = refinement.get("final", {}) or {}

    # Keep report deterministic and readable even if some stages are partial.
    executive_summary = (synthesis.get("executive_summary") or "").strip()
    narrative = (synthesis.get("synthesis") or refined.get("answer") or "").strip()

    lines: list[str] = []
    lines.append(f"# Research Report")
    lines.append("")
    lines.append(f"## Query")
    lines.append(query)
    lines.append("")

    lines.append("## Routing")
    lines.append(f"- Domain: {route.get('domain', 'unknown')}")
    lines.append(f"- Complexity: {route.get('complexity', 'unknown')}")
    try:
        lines.append(f"- Confidence: {float(route.get('confidence', 0.0) or 0.0):.2f}")
    except Exception:
        lines.append("- Confidence: 0.00")
    lines.append("")

    if executive_summary:
        lines.append("## Executive summary")
        lines.append(executive_summary)
        lines.append("")

    lines.append("## Findings")
    lines.append(narrative if narrative else "_No narrative produced._")
    lines.append("")

    lines.append("## Evidence (top sources)")
    if top_sources:
        for i, s in enumerate(top_sources, 1):
            title = (s.get("title") or "Untitled").strip()
            url = (s.get("url") or "").strip()
            if url:
                lines.append(f"{i}. {title} — {url}")
            else:
                lines.append(f"{i}. {title}")
    else:
        lines.append("_No sources available._")
    lines.append("")

    lines.append("## Fact check")
    try:
        lines.append(f"- Credibility score: {float(fact_check.get('credibility_score', 0.0) or 0.0):.2f}")
    except Exception:
        lines.append("- Credibility score: 0.00")
    lines.append(f"- Verified claims: {len(fact_check.get('verified_claims', []) or [])}")
    lines.append(f"- Questionable claims: {len(fact_check.get('questionable_claims', []) or [])}")
    lines.append("")

    lines.append("## Citations (APA)")
    bib = (citations.get("bibliography") or "").strip()
    lines.append(bib if bib else "_No bibliography produced._")
    lines.append("")

    lines.append("## Metrics")
    for k in ("avg_quality", "avg_sources", "avg_iterations", "avg_credibility", "avg_citations", "avg_wall_time_s"):
        if k in metrics:
            lines.append(f"- {k}: {metrics[k]}")
    lines.append("")

    return "\n".join(lines).strip() + "\n"

