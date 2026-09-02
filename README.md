## Google ADK Multi-Agent Research Assistant

Multi-agent research assistant built with **Google Agent Development Kit (ADK)** and **Vertex AI Gemini**. It demonstrates practical agentic workflow patterns—routing, parallel research, evaluator–optimizer refinement, and report generation—using structured JSON contracts between stages.

> **Academic integrity note**: This repository is an independent portfolio implementation demonstrating agentic workflow patterns with Google ADK and Vertex AI. It is **not** intended to be used as a course submission or as a replacement for completing academic work independently.

### Quickstart

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
cp .env.example .env
research-assistant --query "Compare major approaches to retrieval-augmented generation (RAG) for enterprise assistants."
```

### Why it’s interesting

- **Composable workflow graph**: agents are orchestrated as explicit stages with typed handoffs.
- **Parallel research fan-out / fan-in**: multiple “source discovery” workers run concurrently and are aggregated.
- **Evaluator–optimizer loop**: a critic agent scores drafts and can trigger refinement.
- **Validation and attribution**: fact-check pass + citation formatting with deterministic fallbacks.

### Architecture (high level)

```mermaid
flowchart TD
  U[User query] --> O[Workflow Orchestrator]
  O --> R[Query Router]
  O -->|parallel| S1[Source Worker: Web-like]
  O -->|parallel| S2[Source Worker: Papers-like]
  O -->|parallel| S3[Source Worker: Scholar-like]
  S1 --> A[Source Aggregator]
  S2 --> A
  S3 --> A
  A --> L[Refinement Loop (Draft <-> Critic)]
  L --> F[Fact Check]
  F --> Y[Synthesis]
  Y --> C[Citation Formatter]
  C --> M[Metrics + Markdown Report]
```

### Tech stack

- **Python**
- **Google ADK** (`google-adk`)
- **Google GenAI client** (`google-genai`) with **Vertex AI** backend
- **dotenv** configuration

### Setup (detailed)

```bash
git clone <your-repo-url>
cd google-adk-research-assistant

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -e ".[dev]"
```

Alternative (no editable install):

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Create `.env`:

```bash
cp .env.example .env
```

### Environment variables

This project supports:

- **Vertex AI (recommended)**: uses **ADC** (via `gcloud auth application-default login`) or a service-account JSON.
- **API key mode (optional)**: uses `GOOGLE_API_KEY` and does **not** use Vertex AI.

Minimum for Vertex AI:

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
MODEL_NAME=gemini-2.5-flash
```

Vertex AI authentication (ADC):

- `gcloud auth application-default login`
- Windows note: use `gcloud.cmd` if PowerShell script execution is restricted

Optional service-account JSON (otherwise ADC is used):

```env
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

### Running

```bash
python -m research_assistant --query "Compare major approaches to RAG for enterprise knowledge assistants."
```

If you installed the package (recommended), you can also run:

```bash
research-assistant --query "Compare major approaches to RAG for enterprise knowledge assistants."
```

Outputs:
- a Markdown report printed to stdout
- optionally, write to a file via `--out report.md`

### Agentic workflow patterns (where to look)

- **Prompt chaining**: `src/research_assistant/workflows/pipeline.py`
- **Routing**: `src/research_assistant/agents/router.py`
- **Parallelization**: `src/research_assistant/agents/sources.py` (`ParallelAgent`)
- **Evaluator–optimizer**: `src/research_assistant/agents/refinement.py` (`LoopAgent` with a critic threshold)
- **Validation / fact checking**: `src/research_assistant/agents/validation.py`
- **Report generation**: `src/research_assistant/reporting/markdown.py`

### Tests

```bash
pytest
```

### Project structure

```text
src/research_assistant/
  agents/          # LlmAgent implementations (router, sources, critic, etc.)
  workflows/       # orchestration graph + state handoffs
  runtime/         # ADK Runner helpers
  reporting/       # markdown report rendering
  evaluation/      # metrics + summaries
  config/          # settings/env loading
tests/             # portfolio-safe unit tests (no external calls)
docs/              # architecture notes + recommended visuals
```

### Limitations

- “Source discovery” workers are **LLM-simulated** by default (designed to highlight orchestration patterns).
- Fact-checking is model-based; it does not provide deterministic verification.
- Multi-stage pipelines are slower than single-shot generation.

