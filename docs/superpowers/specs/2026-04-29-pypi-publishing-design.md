# PyPI Publishing Pipeline for `osr-metrics` — Design

**Date:** 2026-04-29
**Status:** Approved (design phase)
**Repository:** https://github.com/hxtruong6/osr-metrics
**PyPI project name:** `osr-metrics`

## 1. Goal & Scope

Publish `osr-metrics` to PyPI so end users can `pip install osr-metrics` and reuse the library. Releases are fully automated via GitHub Actions Trusted Publishing (OIDC, no stored tokens).

**Release flow:** bump `version` in `pyproject.toml` → commit → push a `v*` git tag → CI builds, tests, and publishes.

- Pre-release tags (e.g., `v0.1.2rc1`, `v0.1.2-test`, `v0.1.2.dev1`) → **TestPyPI** as a dry run.
- Stable tags matching `^v[0-9]+\.[0-9]+\.[0-9]+$` (e.g., `v0.1.2`) → **PyPI**.

### Non-goals (explicitly deferred)

- ruff / mypy / formatting gates in CI.
- Automated changelog generation or version bumping.
- Conda-forge or any other index.
- Artifact signing (Sigstore, GPG).
- Multi-OS test matrix (Linux only is sufficient for a numpy-only library).

## 2. Pre-flight Cleanup

Repository changes required before the first upload, in priority order.

### 2.1 `pyproject.toml` metadata

- Replace placeholder `Repository` URL with `https://github.com/hxtruong6/osr-metrics`.
- Add the following `[project.urls]` entries:
  - `Homepage = "https://github.com/hxtruong6/osr-metrics"`
  - `Issues = "https://github.com/hxtruong6/osr-metrics/issues"`
  - `Changelog = "https://github.com/hxtruong6/osr-metrics/blob/main/CHANGELOG.md"`
- Add `email` to `authors`.
- Migrate license to SPDX form: `license = "MIT"` and `license-files = ["LICENSE"]`. Drop the legacy `License :: OSI Approved :: MIT License` classifier (deprecated when SPDX `license` is set).

### 2.2 Repo hygiene

- Replace `.gitignore` with the standard Python template (covers `__pycache__/`, `*.egg-info/`, `dist/`, `build/`, `.pytest_cache/`, `.coverage`, `.venv/`, `.mypy_cache/`, etc.).
- Remove tracked build artifacts: `git rm -r --cached osr_metrics.egg-info/ .pytest_cache/` (and any others present).

### 2.3 Package surface

- Audit `osr_metrics/__init__.py` exports against the README's public API surface (`auroc`, `fpr_at_tpr`, `fpr_at_95tpr`, `aupr_in`, `aupr_out`, `compute_aoscr`, `oscr_curve`, `compute_nf_rejection_at_tpr`, `macro_auprc`, `macro_auprc_id_labels`, `macro_f1_with_thresholds`, `per_label_auprc`, `f1_per_label`, `build_fourclass_masks`, `compute_fourclass_metrics`, `partition_ood_by_purity`, `expected_calibration_error`, `brier_score`, `delong_test`, `bootstrap_ci`).
- Add `__version__` to the package, sourced from installed metadata:

  ```python
  from importlib.metadata import version, PackageNotFoundError
  try:
      __version__ = version("osr-metrics")
  except PackageNotFoundError:
      __version__ = "0.0.0+unknown"
  ```

### 2.4 Distribution sanity

- Ensure `LICENSE`, `README.md`, `CHANGELOG.md`, `CITATION.cff` are included in the sdist (verify with `tar tzf dist/osr_metrics-*.tar.gz`). Setuptools includes these by default given the project layout; add a `MANIFEST.in` only if any are missing.
- Run locally:
  - `python -m build` — produces `dist/osr_metrics-<v>.tar.gz` and `dist/osr_metrics-<v>-py3-none-any.whl`.
  - `twine check dist/*` — must report `PASSED` for both files.
  - `pip install dist/osr_metrics-<v>-py3-none-any.whl` in a clean venv; `python -c "from osr_metrics import auroc, __version__; print(__version__)"` must succeed.

## 3. GitHub Actions Workflows

Two workflows under `.github/workflows/`.

### 3.1 `ci.yml` — every push and PR to `main`

```
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

Jobs:

- **`test`** — matrix `python-version: ["3.10", "3.11", "3.12"]`, `runs-on: ubuntu-latest`:
  1. `actions/checkout@v4`
  2. `actions/setup-python@v5` with `cache: pip`
  3. `pip install -e .[dev]`
  4. `pytest tests/ -v`
- **`build`** — single job, `needs: test`:
  1. `pip install build twine`
  2. `python -m build`
  3. `twine check dist/*`
  4. `actions/upload-artifact@v4` with `dist/` (so PR reviewers can inspect the wheel/sdist).

### 3.2 `release.yml` — only on `v*` tags

```
on:
  push:
    tags: ["v*"]

permissions:
  id-token: write   # required for Trusted Publishing OIDC
  contents: read
```

#### Job `verify-and-build`

1. Checkout, `setup-python` 3.11.
2. **Tag/version consistency check:** extract `version` from `pyproject.toml` (via `python -c` or a small inline script) and compare to `${GITHUB_REF_NAME#v}`. Fail with a clear error if they differ.
3. `pip install -e .[dev] && pytest tests/ -v`.
4. `pip install build twine && python -m build && twine check dist/*`.
5. Upload `dist/` as an artifact for the publish jobs.

#### Job `publish-testpypi`

- `needs: verify-and-build`
- `if: contains(github.ref_name, 'rc') || contains(github.ref_name, 'dev') || contains(github.ref_name, '-test') || contains(github.ref_name, 'a') || contains(github.ref_name, 'b')`
  (Rough; final implementation uses a regex check on a non-stable suffix. Equivalent to: any tag that doesn't match `^v[0-9]+\.[0-9]+\.[0-9]+$`.)
- `environment: testpypi`
- Steps:
  1. `actions/download-artifact@v4` to retrieve `dist/`.
  2. `pypa/gh-action-pypi-publish@release/v1` with `repository-url: https://test.pypi.org/legacy/`.

#### Job `publish-pypi`

- `needs: verify-and-build`
- `if: github.ref_name matches ^v[0-9]+\.[0-9]+\.[0-9]+$` (implemented via a small regex check step that sets an output, gating this job).
- `environment: pypi` (with required reviewer for safety).
- Steps:
  1. `actions/download-artifact@v4` to retrieve `dist/`.
  2. `pypa/gh-action-pypi-publish@release/v1` (defaults to PyPI).

### 3.3 Correctness invariants

- **No tokens.** OIDC-only authentication via `pypa/gh-action-pypi-publish`.
- **Build once, publish once.** The publish jobs download the exact artifact built and tested in `verify-and-build` — no rebuild, no drift.
- **Strict stable-tag regex** prevents `v0.1.2rc1` from accidentally hitting PyPI.

## 4. PyPI / TestPyPI Account Setup

Done by the maintainer in a browser before the first release. *(Confirmed complete as of 2026-04-29.)*

### 4.1 Account prep

- PyPI account with 2FA enabled and recovery codes saved.
- TestPyPI account with 2FA enabled.

### 4.2 Pending Trusted Publishers

Configured on **PyPI** (account → Publishing → Add pending publisher):

| Field | Value |
|---|---|
| PyPI Project Name | `osr-metrics` |
| Owner | `hxtruong6` |
| Repository name | `osr-metrics` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Same on **TestPyPI**, with **Environment name: `testpypi`**.

### 4.3 GitHub Environments

In repo Settings → Environments:

- `pypi` — required reviewer (the maintainer); optional tag protection rule limiting to `v*` tags.
- `testpypi` — no reviewer required.

Environment names must match the pending-publisher configurations exactly.

### 4.4 First release dry-run order

1. Tag `v0.1.2rc1` → workflow → publishes to **TestPyPI**.
2. Manual verification on TestPyPI:
   - Project page renders the README and sidebar links.
   - `pip install -i https://test.pypi.org/simple/ osr-metrics==0.1.2rc1` works in a clean venv.
   - `python -c "from osr_metrics import auroc, __version__; print(__version__)"` prints `0.1.2rc1`.
3. Tag `v0.1.2` → workflow → environment approval → publishes to **PyPI**.

After the first successful upload to each index, the pending publisher is promoted to a regular publisher automatically.

## 5. Documentation Updates

### 5.1 `README.md`

- Change the primary install instruction from `pip install -e .` to `pip install osr-metrics`. Keep editable install under a "Development" subsection.
- Add a badge row at the top:
  - PyPI version (shields.io `pypi/v/osr-metrics`)
  - Python versions (`pypi/pyversions/osr-metrics`)
  - License (`pypi/l/osr-metrics`)
  - CI status (GitHub Actions badge for `ci.yml`)

### 5.2 `CHANGELOG.md`

Add an entry:

```
## [0.1.2] - 2026-04-29
### Added
- PyPI distribution; `pip install osr-metrics`.
- GitHub Actions CI (pytest matrix on Python 3.10/3.11/3.12, build, twine check).
- Release workflow with Trusted Publishing (TestPyPI for pre-release tags, PyPI for stable).
- `osr_metrics.__version__` sourced from installed metadata.

### Changed
- `pyproject.toml` metadata: real Repository/Issues/Changelog URLs, SPDX license.

### Removed
- Tracked build artifacts (`*.egg-info/`, `.pytest_cache/`).
```

### 5.3 `docs/RELEASING.md` (new)

A short checklist for future releases (~30 lines): version bump → CHANGELOG entry → commit → rc tag → TestPyPI verify → stable tag → PyPI verify. Lives alongside `docs/USAGE.md` and `docs/EXAMPLES.md`.

## 6. Acceptance Criteria

The pipeline is shipped when **all** of the following hold:

1. `ci.yml` runs green on a PR opened against `main`: pytest passes on Python 3.10, 3.11, 3.12; `python -m build` succeeds; `twine check dist/*` reports `PASSED`.
2. Tagging `v0.1.2rc1` triggers `release.yml`, which:
   - Confirms the tag matches `pyproject.toml` version.
   - Builds and publishes to TestPyPI via OIDC.
   - `https://test.pypi.org/project/osr-metrics/0.1.2rc1/` is live, README renders, sidebar links work.
   - `pip install -i https://test.pypi.org/simple/ osr-metrics==0.1.2rc1` succeeds in a clean venv.
3. Tagging `v0.1.2` triggers the same workflow, prompts for `pypi` environment approval, and publishes to PyPI.
   - `https://pypi.org/project/osr-metrics/0.1.2/` is live with correct metadata.
   - `pip install osr-metrics` in a clean venv installs and imports cleanly.
4. No API tokens stored anywhere — not in workflows, GitHub Secrets, local `.pypirc`, nor environment variables.
5. `osr_metrics.__version__` returns `"0.1.2"` after install.
6. Working tree is clean: no `*.egg-info/`, `dist/`, `build/`, `__pycache__/`, or `.pytest_cache/` tracked in git.

## 7. Open Risks

- **README rendering on PyPI** — PyPI uses a stricter Markdown renderer than GitHub. Mitigation: `twine check` in CI catches metadata-level rendering failures; the rc1 → TestPyPI dry-run catches anything subtler.
- **Trusted-publisher misconfig** — environment-name mismatch between GitHub and PyPI silently fails OIDC. Mitigation: first failure is on the rc1 push, before any real PyPI version is consumed.
- **Name collision** — verified that `osr-metrics` was free at design time. If someone else claims it before the first upload, fall back to `osr-metrics-py` (matches the GitHub repo of the source folder name).
