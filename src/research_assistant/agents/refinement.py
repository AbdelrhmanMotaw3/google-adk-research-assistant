from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from google import genai
from google.adk.agents import LlmAgent, LoopAgent
from google.genai.types import GenerateContentConfig

from research_assistant.config.settings import Settings
from research_assistant.runtime.adk_runner import event_text, run_agent_events
from research_assistant.utils.json_parsing import parse_jsonish


class DraftAgent(LlmAgent):
    def __init__(self, model: str):
        instruction = """You are a research assistant.

You will receive a research question and a JSON list of candidate sources.
Ground key claims in those sources when possible (do not invent sources).

Return STRICT JSON:
{
  "answer": "2-6 paragraphs",
  "key_points": ["...", "...", "..."],
  "sources_mentioned": ["title or url", "..."],
  "confidence": "high|medium|low"
}
Return ONLY JSON."""

        super().__init__(
            name="draft_writer",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1200,
                response_mime_type="application/json",
            ),
        )


class CriticAgent(LlmAgent):
    def __init__(self, model: str, *, quality_threshold: float):
        self._threshold = float(quality_threshold)
        instruction = """You are a research quality critic.

Score the draft for:
- accuracy and clarity
- coverage of the question
- grounding in provided sources

Return STRICT JSON:
{
  "quality_score": 0.0-1.0,
  "verdict": "accept|revise",
  "feedback": "actionable feedback in 1-4 sentences"
}

Set verdict="accept" only if quality_score is high enough.
Return ONLY JSON."""

        def _after_model_set_escalate(callback_context, llm_response):
            try:
                txt = ""
                if llm_response.content and llm_response.content.parts:
                    txt = "".join([p.text for p in llm_response.content.parts if getattr(p, "text", None)]).strip()
                parsed = parse_jsonish(txt) if txt else {}
                score = float(parsed.get("quality_score", 0.0) or 0.0) if isinstance(parsed, dict) else 0.0
                if score >= self._threshold:
                    callback_context.actions.escalate = True
            except Exception:
                return None
            return None

        super().__init__(
            name="draft_critic",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
            after_model_callback=_after_model_set_escalate,
        )


async def refine_answer(
    *,
    client: genai.Client,
    settings: Settings,
    query: str,
    sources: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    draft = DraftAgent(model=model)
    critic = CriticAgent(model=model, quality_threshold=settings.quality_threshold)

    loop = LoopAgent(name="refinement_loop", sub_agents=[draft, critic], max_iterations=settings.max_refinement_iters)

    message = (
        "Research question:\n"
        f"{query}\n\n"
        "SOURCES_JSON:\n"
        f"{json.dumps(sources, indent=2) if sources else '[]'}"
    )

    events = await run_agent_events(loop, message=message, settings=settings, app_name="refinement")

    answers: List[Dict[str, Any]] = []
    critiques: List[Dict[str, Any]] = []
    for ev in events:
        if not ev.is_final_response():
            continue
        txt = event_text(ev)
        if not txt:
            continue
        if ev.author == draft.name:
            try:
                parsed = parse_jsonish(txt)
                answers.append(parsed if isinstance(parsed, dict) else {"answer": json.dumps(parsed)})
            except Exception:
                answers.append({"answer": txt, "key_points": [], "sources_mentioned": [], "confidence": "low"})
        elif ev.author == critic.name:
            try:
                parsed = parse_jsonish(txt)
                critiques.append(parsed if isinstance(parsed, dict) else {"quality_score": 0.0, "verdict": "revise", "feedback": "Invalid critique payload"})
            except Exception:
                critiques.append({"quality_score": 0.0, "verdict": "revise", "feedback": "Unable to parse critique"})

    history: List[Dict[str, Any]] = []
    for i in range(min(len(answers), len(critiques))):
        history.append({"iteration": i + 1, "draft": answers[i], "critique": critiques[i]})

    def _score(item: Dict[str, Any]) -> float:
        try:
            return float((item.get("critique") or {}).get("quality_score", 0.0) or 0.0)
        except Exception:
            return 0.0

    best = max(history, key=_score) if history else None
    final_draft = (best or history[-1])["draft"] if history else {}

    return {
        "final": final_draft,
        "history": history,
        "iterations": len(history),
        "best_iteration": (best or {}).get("iteration") if best else None,
        "loop_agent_name": loop.name,
    }

