# coding_standard.md

## Core Rules
- **Reuse before you write**: Search `./llm_utils` for needed functionality; import it when found.  
  - If functionality is useful to many projects, propose adding it to `llm_utils` (never edit it directly).
  - More instructions on the package contents of llm_utils is available in llm_utils_guide.md file
- Use `aiweb_common` classes/methods for LLM calls.
- Target Python ≥ 3.11; always annotate types.
- Keep functions small, pure, and idempotent when possible.
- Organize code:
  - `ML_Project/` main package & sub-packages
  - `tests/` pytest suite
  - `docs/` MkDocs material
  - `docs/uml/` PlantUML sources
- Protect PHI/PII: no secrets or patient data in code or logs.
- Package Management:
  - Python and corresponding support packages are installed in `.venv/`
  - uv is preferred packagemenent platform
  - This is pre-installed in the development environement through Docker.

## CI Expectations
- PR must pass: unit tests, `black`, `isort`, `autopep8`, `mypy`, `pytest -q`.
- Add/adjust tests with every behavioral change.
