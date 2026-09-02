from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    # repo_root/src/research_assistant/config/settings.py -> repo_root
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Settings:
    gcp_project: str
    gcp_location: str
    model_name: str
    max_refinement_iters: int
    quality_threshold: float
    google_api_key: str | None
    credentials_path: Path | None

    @property
    def using_vertex(self) -> bool:
        return bool(self.gcp_project and self.gcp_location and not self.google_api_key)


def load_settings() -> Settings:
    """
    Load configuration from `.env` + environment variables.

    This function intentionally avoids printing environment values to reduce
    the risk of leaking secrets in logs.
    """
    root = _project_root()
    load_dotenv(root / ".env")

    # Prefer GOOGLE_CLOUD_* but accept PROJECT_ID/LOCATION for compatibility.
    gcp_project = (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID") or "").strip()
    gcp_location = (os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("LOCATION") or "us-central1").strip()
    model_name = (os.getenv("MODEL_NAME") or "gemini-2.5-flash").strip()

    max_refinement_iters = int(os.getenv("MAX_REFINEMENT_ITERS") or os.getenv("MAX_ITERATIONS") or "3")
    quality_threshold = float(os.getenv("QUALITY_THRESHOLD") or "0.80")

    google_api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip() or None

    cred_raw = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    credentials_path = None
    if cred_raw:
        p = Path(cred_raw)
        credentials_path = (root / p).resolve() if not p.is_absolute() else p

    # Ensure google-auth finds the key file when provided.
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    else:
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    return Settings(
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        model_name=model_name,
        max_refinement_iters=max_refinement_iters,
        quality_threshold=quality_threshold,
        google_api_key=google_api_key,
        credentials_path=credentials_path,
    )

