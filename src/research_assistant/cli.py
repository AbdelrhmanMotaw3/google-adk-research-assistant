from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from research_assistant.config.settings import load_settings
from research_assistant.reporting.markdown import render_markdown_report
from research_assistant.workflows.pipeline import run_research_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Google ADK multi-agent research assistant")
    p.add_argument("--query", required=True, help="Research question to investigate")
    p.add_argument("--out", default="", help="Optional path to write Markdown report")
    return p


async def _run(query: str, out: str) -> int:
    settings = load_settings()
    results = await run_research_pipeline(query=query, settings=settings)
    md = render_markdown_report(results)

    if out:
        out_path = Path(out)
        out_path.write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args.query, args.out))

