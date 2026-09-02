from __future__ import annotations

from google import genai

from research_assistant.config.settings import Settings


def create_client(settings: Settings) -> genai.Client:
    """
    Create a google-genai Client configured for Vertex AI or API key auth.

    - If GOOGLE_API_KEY/GEMINI_API_KEY is set: uses API key mode (non-Vertex).
    - Otherwise: uses Vertex AI mode and relies on ADC or service-account JSON.
    """
    if settings.google_api_key:
        return genai.Client(api_key=settings.google_api_key)

    if not settings.gcp_project:
        raise RuntimeError(
            "Missing GOOGLE_CLOUD_PROJECT (or PROJECT_ID). "
            "Set it in your environment or .env file."
        )

    return genai.Client(vertexai=True, project=settings.gcp_project, location=settings.gcp_location)

