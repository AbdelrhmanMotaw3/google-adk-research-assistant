from __future__ import annotations

import os
from typing import Optional, List
from uuid import uuid4

from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from research_assistant.config.settings import Settings


def _make_user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def event_text(event: Event) -> str:
    if not event.content or not event.content.parts:
        return ""
    chunks: list[str] = []
    for part in event.content.parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks).strip()


def last_final_text(events: List[Event], *, author: Optional[str] = None) -> str:
    for ev in reversed(events):
        if author and ev.author != author:
            continue
        if ev.is_final_response():
            txt = event_text(ev)
            if txt:
                return txt
    return ""


async def run_agent_events(agent, *, message: str, settings: Settings, app_name: str = "research_assistant") -> List[Event]:
    """
    Execute an ADK agent through Runner and return all events.

    Notes:
    - ADK constructs google-genai clients internally.
    - For Vertex AI mode, google-genai switches to Vertex when GOOGLE_GENAI_USE_VERTEXAI=true
      and project/location are set.
    """
    if not settings.google_api_key and settings.gcp_project:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.gcp_project)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.gcp_location)

    session_service = InMemorySessionService()
    runner = Runner(app_name=app_name, agent=agent, session_service=session_service, auto_create_session=True)

    events: List[Event] = []
    session_id = uuid4().hex
    async for event in runner.run_async(
        user_id="local_user",
        session_id=session_id,
        new_message=_make_user_content(message),
    ):
        events.append(event)

    return events

