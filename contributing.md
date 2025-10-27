# Contributing

Thanks for helping push LightDiffusion-Next forward! This project blends a Streamlit UI, a Gradio deployment surface, a FastAPI queue and a sizeable inference stack. The guidelines below should get you productive quickly.

## Getting your environment ready

### Prerequisites

- Python 3.10 (the bundled wheels and Stable-Fast extension are built against 3.10)
- NVIDIA GPU with CUDA 12.1+ drivers (for GPU development)
- Git (with LFS if you plan to version large model weights)
- `uv` or `pip` for dependency management
- Optional: Docker + NVIDIA Container Toolkit for containerized testing

### Clone & install

```fish
git clone https://github.com/Aatricks/LightDiffusion-Next.git
cd LightDiffusion-Next

# Recommended: isolate dependencies
python -m venv .venv
source .venv/bin/activate

# Install runtime dependencies
uv pip install -r requirements.txt

# (Optional) Extras for docs and linting
uv pip install mkdocs mkdocs-material mkdocstrings-python ruff black
```

Populate `include/` with the checkpoints you need (SD1.5, Flux, LoRAs, embeddings). The UI will prompt you for missing assets if you skip this step.

## Running the apps locally

- **Streamlit UI**: `streamlit run streamlit_app.py`
- **Gradio UI**: `python app.py`
- **FastAPI backend**: `uvicorn server:app --host 0.0.0.0 --port 7861`

All services read the same configuration and model directories. When working on the pipeline, it’s handy to keep FastAPI running for quick REST smoke tests while you iterate on the UI in a separate terminal.

## Workflow expectations

1. Create a branch per piece of work: `git checkout -b feature/short-summary`.
2. Keep pull requests focused—avoid bundling unrelated refactors with feature work.
3. Reference issues in your commit messages and PR description when applicable.
4. Update documentation (`docs/`, `README.md`) whenever behavior, defaults or environments change.

## Coding standards

- Follow PEP 8 for Python. If you have `ruff` or `black` installed, run them before committing (`ruff check src ui` and `black src ui`).
- Prefer type hints for new modules; FastAPI schemas and pipeline helpers already use Pydantic models you can extend.
- Favor dependency injection over global state—pass configuration into functions where feasible so the FastAPI worker and Streamlit UI stay in sync.
- When touching CUDA or kernel build logic, document the change in `docs/quirks.md` or `docs/installation.md` so operators know about new requirements.

## Verification checklist

Before opening a pull request:

- [ ] `streamlit run streamlit_app.py` starts without stack traces.
- [ ] `uvicorn server:app --host 0.0.0.0 --port 7861` accepts at least one `/api/generate` call (you can use the example payload in [API docs](api.md)).
- [ ] `python app.py` (Gradio) loads when relevant to your change.
- [ ] `mkdocs build` succeeds (documentation stays green).
- [ ] GPU-specific changes are tested on at least one real GPU and noted in the PR description.
- [ ] No large binaries or secrets are committed—place models inside `include/` and gitignore keeps them local.

If you add scripts or automation, include instructions in `docs/examples.md` or a new page and wire it into `mkdocs.yml`.

## Submitting your PR

- Fill out a concise description covering **what changed**, **why**, and any ops impact (new env vars, caches, etc.).
- Attach screenshots or sample renders when altering the UI or pipeline defaults.
- Expect friendly but thorough reviews—batching, caching and GPU tweaks affect many users, so be ready to iterate.
- Squash-merge is fine, but avoid force-pushing after reviews unless you coordinate with the maintainer.

## Bug reports & feature requests

When reporting an issue, please include:

- Operating system, driver versions (`nvidia-smi` output), GPU model
- How you launched LightDiffusion-Next (Streamlit, Docker, FastAPI)
- Relevant logs (`logs/server.log`, Streamlit terminal output, `/api/telemetry` response)
- Steps to reproduce and whether the problem is reproducible on a fresh checkout

Feature ideas are welcome—outline the use case, expected UX and any new dependencies (models, GPU requirements). Discussions and prototypes in separate branches make reviews easier.

## Documentation contributions

- Run `mkdocs serve` while editing to preview changes at http://127.0.0.1:8080.
- Add new pages under `docs/` and update `mkdocs.yml` navigation.
- Screenshots should be optimized PNGs or WebPs stored under `docs/images/`.
- Keep `README.md` focused on quick start—you can link to richer docs pages for details.

Thanks again for contributing! 🚀