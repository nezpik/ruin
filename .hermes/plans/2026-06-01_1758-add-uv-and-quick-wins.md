# RUIN v0.2 — Add uv + Quick Wins Plan

**Date:** 2026-06-01  
**Author:** Hermes (virtual co-founder)  
**Status:** READY FOR EXECUTION (user approval)  
**Process adherence:** Same v0.2 Foundation Hardening flow (small TDD bites, golden config verification, commit + full test suite after each verified phase)

---

## Goal

1. Migrate RUIN to `uv` as the primary package / environment manager (lockfile, reproducible installs, modern Python packaging).
2. Capture low-risk, high-value **quick wins** discovered during analysis before the project grows (logging, .gitignore, pre-commit skeleton, docs alignment).
3. Preserve 100% backward compatibility with existing `ruin simulate` / `ruin risk` commands and the golden `examples/urban_ruin_shift.yaml`.

---

## Current State (Analysis Performed)

**pyproject.toml**
- Uses legacy `setuptools` backend.
- No lockfile → non-reproducible installs.
- Dev/test extras defined but no `ruff format`, pre-commit, or CI enforcement.

**Source code**
- `ruin/cli.py` still uses raw `print()` for all user output (potential structured logging opportunity).
- `ruin/risk/ruin_probability.py` imports `os` (likely for `os.cpu_count` — acceptable for now).
- No existing logging configuration.

**Repository hygiene**
- `.gitignore` misses `uv.lock`, `.ruff_cache/`, `site/`, common IDE noise.
- `config_ex.md` is out of sync with the v0.2 Pydantic model (still shows v0.1 nested dict shape).

**Testing / reproducibility**
- Strong test coverage already (19+ tests).
- Golden config exists and is the official verification artifact.

**No CI yet** — future phase.

---

## Proposed Approach

### Phase 0 — Prerequisites (no code change)
- Verify current tests + golden run still pass on the host Python.
- Document baseline: `ruin simulate --config examples/urban_ruin_shift.yaml --max-steps 20 --visualize` output + GIF hash.

### Phase 1 — uv Migration (core change)
1. Install `uv` in the environment.
2. Initialize `uv` project (or manually create `uv.toml` + `uv.lock`).
3. Convert `[project]` + extras to uv-native format (keep `pyproject.toml` as single source of truth).
4. Generate `uv.lock`.
5. Update developer docs + README quickstart to prefer `uv sync` / `uv run`.
6. Add `uv.lock` to git.

**Verification gate:** Golden config run produces identical text frames + GIF to baseline.

### Phase 2 — Quick Wins (parallelizable, low-risk)
- **Logging quick-win** — Replace `print()` in `cli.py` with `logging.getLogger(__name__)` + basic config (still outputs to stdout for CLI users).  
- **.gitignore hardening** — Add `uv.lock` (intentional), `.ruff_cache/`, `.mypy_cache/`, `site/`, `.idea/`, `.vscode/`.
- **Pre-commit skeleton** — Add minimal `.pre-commit-config.yaml` (ruff, mypy, trailing-whitespace) + `pre-commit` dev dependency. (No enforcement yet.)
- **Docs alignment** — Update `config_ex.md` header to note “v0.2 Pydantic model supersedes the example YAML shown here”.

**Verification gate:** All tests still green; golden run unchanged; new files committed.

### Phase 3 — Post-merge polish (future, not in this plan)
- Add GitHub Action that runs `uv run pytest` + golden config.
- Consider `uv tool install` support for the `ruin` CLI.
- Evaluate `uv build` + PyPI publish workflow.

---

## Step-by-Step Execution Plan (Bite-sized, Same Process)

### Phase 0 — Baseline Verification (≈15 min)
1. `cd /opt/data/ruin`
2. `python -m pip install -e ".[test]"` (or equivalent)
3. `pytest -q --tb=no`
4. `ruin simulate --config examples/urban_ruin_shift.yaml --max-steps 20 --output /tmp/baseline.txt --visualize`
5. Record SHA256 of `/tmp/baseline.txt` and the produced GIF.
6. Commit nothing — just capture baseline.

**Files touched:** none (verification only)

### Phase 1 — uv Migration
1. `uv --version` (install if missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. `uv init --app --package` (or manual edit)
3. Edit `pyproject.toml`:
   - Change build backend to `hatchling` or keep `setuptools` (uv supports both).
   - Add `[tool.uv]` section with `dev-dependencies`.
4. `uv sync --dev`
5. `uv lock`
6. Update README quickstart:
   ```bash
   uv sync --dev
   uv run ruin simulate --config examples/urban_ruin_shift.yaml ...
   ```
7. Add `uv.lock` + any new `.gitignore` entries.
8. Commit: `git add pyproject.toml uv.lock .gitignore README.md && git commit -m "chore: migrate to uv + lockfile (Phase 1)"`

**Verification:** Re-run golden config; assert identical output to baseline.

**Files likely changed:**
- `pyproject.toml`
- `uv.lock` (new)
- `README.md`
- `.gitignore`

### Phase 2 — Quick Wins (can be one or more commits)

**2a — Structured logging in CLI**
- In `ruin/cli.py`:
  - Add `import logging`
  - Replace `print(json.dumps(...))` with `logging.getLogger("ruin.cli").info(...)`
  - Add `logging.basicConfig(level=logging.INFO, format="%(message)s")` at top of `main()`
- Keep exact same stdout JSON for downstream scripts.

**2b — .gitignore update**
- Append:
  ```
  uv.lock
  .ruff_cache/
  .mypy_cache/
  site/
  .idea/
  .vscode/
  ```

**2c — Pre-commit skeleton (optional but high-leverage)**
- `uv add --dev pre-commit`
- Create `.pre-commit-config.yaml` with:
  ```yaml
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      hooks: [ruff, ruff-format]
    - repo: https://github.com/pre-commit/mirrors-mypy
      hooks: [mypy]
  ```
- Document in README under “Development”.

**2d — Docs alignment**
- In `config_ex.md`: add note at top:
  > **Note (v0.2):** The authoritative configuration is defined in `ruin/config_models.py` (Pydantic). The YAML below is illustrative only.

**Commit example:**
```
git add ruin/cli.py .gitignore .pre-commit-config.yaml config_ex.md pyproject.toml
git commit -m "chore: quick wins — logging, gitignore, pre-commit skeleton, docs (Phase 2)"
```

**Verification:** Full test suite + golden config run.

---

## Risk / Trade-offs / Open Questions

| Item                    | Risk Level | Mitigation                                      |
|-------------------------|------------|-------------------------------------------------|
| uv adoption friction    | Low        | Keep `pip install -e .` working as fallback     |
| Logging format change   | Very Low   | Use `%(message)s` so JSON output is identical   |
| Pre-commit is optional  | None       | Not enforced until CI phase                     |
| `uv.lock` committed     | None       | Standard practice for reproducibility           |
| config_ex.md staleness  | Low        | Only header comment — no content change         |

**Open questions for user:**
- Do we want `hatchling` (modern) or keep `setuptools` during uv migration?
- Should `uv tool install nezpik/ruin` be a goal (makes `ruin` globally available)?
- Any objection to adding the pre-commit skeleton now?

---

## Success Criteria (Definition of Done for this plan)

- `uv sync --dev` produces a working environment.
- `uv run ruin simulate ...` and `uv run ruin risk ...` work identically to current behavior.
- Golden config run produces byte-for-byte identical text output and GIF as baseline.
- All 19+ tests still pass.
- `uv.lock` is committed.
- Quick-win changes are in a separate, cleanly-reviewable commit.
- Plan file lives at `.hermes/plans/2026-06-01_1758-add-uv-and-quick-wins.md`.

---

**Next action (if approved):** Execute Phase 0 baseline verification, then Phase 1 in a single focused session with “Continue” checkpoints after each gate.