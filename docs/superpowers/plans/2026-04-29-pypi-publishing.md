# PyPI Publishing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `osr-metrics` to PyPI via GitHub Actions Trusted Publishing, so `pip install osr-metrics` works for end users.

**Architecture:** Two GitHub Actions workflows (`ci.yml` for tests/build on push/PR; `release.yml` on `v*` tags). Tag-driven releases route pre-release tags (e.g., `v0.1.2rc1`) to TestPyPI and stable tags (`v0.1.2`) to PyPI. No tokens stored anywhere — OIDC Trusted Publishing only. Build-once / publish-once: the artifact tested in `verify-and-build` is the artifact uploaded.

**Tech Stack:** Python 3.10/3.11/3.12, `setuptools` build backend, `build`, `twine`, GitHub Actions, `pypa/gh-action-pypi-publish@release/v1`.

**Spec reference:** `docs/superpowers/specs/2026-04-29-pypi-publishing-design.md`.

**Pre-existing setup (already done by maintainer):**
- PyPI + TestPyPI accounts with 2FA.
- Pending Trusted Publishers configured on both indexes for project `osr-metrics`, repo `hxtruong6/osr-metrics`, workflow `release.yml`, environments `pypi` and `testpypi`.
- GitHub Environments `pypi` (with required reviewer) and `testpypi` (no reviewer) created in the repo.

---

## File Structure

**Files this plan creates:**
- `.github/workflows/ci.yml` — pytest matrix + build + twine check on push/PR.
- `.github/workflows/release.yml` — tag-driven build + publish to TestPyPI or PyPI.
- `docs/RELEASING.md` — maintainer checklist for cutting future releases.

**Files this plan modifies:**
- `pyproject.toml` — URLs, author email, SPDX license, version bump to 0.1.2.
- `osr_metrics/__init__.py` — switch `__version__` to `importlib.metadata`.
- `README.md` — badges row, primary install instruction.
- `CHANGELOG.md` — 0.1.2 entry.
- `.gitignore` — already covers the relevant patterns; no changes needed.

**Files this plan removes from git tracking (keeps on disk, ignored going forward):**
- `osr_metrics.egg-info/`
- `.pytest_cache/`

---

## Task 1: Remove Tracked Build Artifacts

**Files:**
- Untrack: `osr_metrics.egg-info/`, `.pytest_cache/`

- [ ] **Step 1: Confirm what's currently tracked**

Run: `git ls-files | grep -E '(egg-info|pytest_cache)' | head`
Expected: lists files inside `osr_metrics.egg-info/` and/or `.pytest_cache/`.

- [ ] **Step 2: Untrack them (keep local copies)**

Run:
```bash
git rm -r --cached osr_metrics.egg-info/ .pytest_cache/ 2>/dev/null || true
```
Expected: prints `rm '...'` lines for each file, or no-ops silently if a directory doesn't exist.

- [ ] **Step 3: Verify nothing else suspicious is tracked**

Run: `git ls-files | grep -E '(__pycache__|\.pyc$|/dist/|/build/)' || echo CLEAN`
Expected: prints `CLEAN`.

- [ ] **Step 4: Verify `.gitignore` covers the relevant patterns**

Read `.gitignore`. It must contain `__pycache__/`, `*.egg-info/`, `dist/`, `build/`, `.pytest_cache/`. The current file already does — no edit required. If any are missing, append them.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: untrack build artifacts (egg-info, pytest_cache)"
```

---

## Task 2: Update `pyproject.toml` Metadata

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace project metadata**

In `pyproject.toml`, replace the entire `[project]` block (lines 5-30) with:

```toml
[project]
name = "osr-metrics"
version = "0.1.1"
description = "Open-Set Recognition (OSR) and OOD-detection metrics for ML research"
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
requires-python = ">=3.10"
authors = [
    { name = "truong dev", email = "hxtruong6ac@gmail.com" }
]
keywords = ["open-set-recognition", "ood-detection", "metrics", "auroc", "aoscr", "delong", "calibration"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Intended Audience :: Science/Research",
]
dependencies = [
    "numpy>=1.23",
    "scikit-learn>=1.2",
    "scipy>=1.10",
]
```

Notes on what changed:
- Added `email` to `authors`.
- `license` is now an SPDX string (`"MIT"`) and `license-files = ["LICENSE"]` replaces the legacy `license = { file = "LICENSE" }` form.
- Removed the `License :: OSI Approved :: MIT License` classifier (PyPI deprecates it when SPDX `license` is set).
- Version stays at `0.1.1` — Task 5 bumps it.

- [ ] **Step 2: Replace `[project.urls]` with the real URLs**

Replace the existing `[project.urls]` block:

```toml
[project.urls]
Homepage = "https://github.com/hxtruong6/osr-metrics"
Repository = "https://github.com/hxtruong6/osr-metrics"
Issues = "https://github.com/hxtruong6/osr-metrics/issues"
Changelog = "https://github.com/hxtruong6/osr-metrics/blob/main/CHANGELOG.md"
```

- [ ] **Step 3: Verify the file parses**

Run:
```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(pyproject): real URLs, author email, SPDX license"
```

---

## Task 3: Source `__version__` from Installed Metadata

**Files:**
- Modify: `osr_metrics/__init__.py:31`
- Test: `tests/test_version.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_version.py`:

```python
"""Tests for osr_metrics.__version__ wiring."""
import re

import osr_metrics


def test_version_is_string():
    assert isinstance(osr_metrics.__version__, str)
    assert osr_metrics.__version__  # non-empty


def test_version_matches_pyproject():
    """__version__ must equal the version declared in pyproject.toml.

    When the package is installed (editable or wheel), importlib.metadata
    returns the version setuptools recorded at install time. Mismatch means
    someone bumped pyproject.toml without reinstalling.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert osr_metrics.__version__ == declared, (
        f"__version__={osr_metrics.__version__!r} but pyproject declares "
        f"{declared!r}. Reinstall with `pip install -e .` after bumping."
    )


def test_version_looks_like_pep440():
    # Loose check: digits.digits.digits with optional pre/dev/post suffix.
    assert re.match(r"^\d+\.\d+\.\d+", osr_metrics.__version__)
```

- [ ] **Step 2: Run the test to verify it currently passes (sanity)**

Run: `pytest tests/test_version.py -v`
Expected: PASS — `__version__` is hardcoded `"0.1.1"` and `pyproject.toml` declares `0.1.1`. The test is here to catch *future* drift after we move to `importlib.metadata`.

- [ ] **Step 3: Modify `osr_metrics/__init__.py` to use `importlib.metadata`**

Replace line 31 (`__version__ = "0.1.1"`) with:

```python
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("osr-metrics")
except PackageNotFoundError:  # not installed (e.g. running from source tree without install)
    __version__ = "0.0.0+unknown"
```

Place this block immediately above the existing `__all__ = [...]` declaration. Leave the rest of the file alone.

- [ ] **Step 4: Reinstall the package so metadata is current**

Run: `pip install -e .`
Expected: `Successfully installed osr-metrics-0.1.1`.

- [ ] **Step 5: Run the version tests**

Run: `pytest tests/test_version.py -v`
Expected: 3 tests PASS.

- [ ] **Step 6: Run the full suite to confirm nothing else broke**

Run: `pytest tests/ -v`
Expected: all tests pass (existing tests + the 3 new ones).

- [ ] **Step 7: Commit**

```bash
git add osr_metrics/__init__.py tests/test_version.py
git commit -m "feat: source __version__ from installed metadata"
```

---

## Task 4: Local Build Dry-Run

This task verifies the cleaned-up metadata produces a valid package *before* we wire CI. Catches problems where they're cheapest to fix.

**Files:** none modified — verification only.

- [ ] **Step 1: Install build tools**

Run: `pip install --upgrade build twine`
Expected: `Successfully installed build-... twine-...`.

- [ ] **Step 2: Clean any stale artifacts**

Run: `rm -rf dist/ build/ osr_metrics.egg-info/`
Expected: silent success.

- [ ] **Step 3: Build sdist + wheel**

Run: `python -m build`
Expected: ends with `Successfully built osr_metrics-0.1.1.tar.gz and osr_metrics-0.1.1-py3-none-any.whl`.

- [ ] **Step 4: Twine check (validates README rendering for PyPI)**

Run: `twine check dist/*`
Expected:
```
Checking dist/osr_metrics-0.1.1-py3-none-any.whl: PASSED
Checking dist/osr_metrics-0.1.1.tar.gz: PASSED
```
If `FAILED`: read the error message, fix the offending markup in `README.md`, repeat from Step 2.

- [ ] **Step 5: Inspect sdist contents**

Run: `tar tzf dist/osr_metrics-0.1.1.tar.gz | sort`
Expected: includes `LICENSE`, `README.md`, `pyproject.toml`, `osr_metrics/*.py`. CHANGELOG and CITATION inclusion is nice-to-have but not required (they live at the repo root and setuptools may or may not pick them up). If they're missing and you want them in the sdist, create `MANIFEST.in` with:

```
include CHANGELOG.md
include CITATION.cff
```

Then rebuild and re-verify.

- [ ] **Step 6: Smoke-test the wheel in a clean venv**

Run:
```bash
python -m venv /tmp/osr-test-venv
/tmp/osr-test-venv/bin/pip install dist/osr_metrics-0.1.1-py3-none-any.whl
/tmp/osr-test-venv/bin/python -c "from osr_metrics import auroc, __version__; print(__version__)"
```
Expected: prints `0.1.1`.

- [ ] **Step 7: Clean up**

Run:
```bash
rm -rf /tmp/osr-test-venv dist/ build/
```

- [ ] **Step 8: Commit (only if MANIFEST.in was added in step 5)**

If you created `MANIFEST.in`:

```bash
git add MANIFEST.in
git commit -m "chore: include CHANGELOG and CITATION in sdist"
```

Otherwise nothing to commit; this task was verification-only.

---

## Task 5: Bump Version to 0.1.2 and Update Changelog

**Files:**
- Modify: `pyproject.toml` (version field)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version in `pyproject.toml`**

Change `version = "0.1.1"` to `version = "0.1.2"`.

- [ ] **Step 2: Update `CHANGELOG.md`**

Replace the current `## [Unreleased]` section with the following two sections (the existing `[Unreleased]` content from the prior commit becomes part of `[0.1.2]`):

```markdown
## [Unreleased]

## [0.1.2] — 2026-04-29

### Added
- PyPI distribution: `pip install osr-metrics`.
- GitHub Actions CI: pytest matrix on Python 3.10/3.11/3.12, `python -m build`, `twine check`.
- Tag-driven release workflow with Trusted Publishing (OIDC). Pre-release tags
  (`vX.Y.ZrcN`) publish to TestPyPI; stable tags (`vX.Y.Z`) publish to PyPI.
- `osr_metrics.__version__` now sourced from installed metadata via
  `importlib.metadata` (was a hardcoded string).
- `docs/RELEASING.md` — maintainer checklist for cutting releases.
- `docs/USAGE.md` — decision-tree guide for picking the right metric.
- `docs/EXAMPLES.md` — end-to-end runnable example covering the full
  publication metric panel, DeLong comparison, and seed aggregation.
- `CHANGELOG.md` (this file).
- `CITATION.cff` — machine-readable citation metadata.

### Changed
- `pyproject.toml` metadata: real Repository/Issues/Changelog URLs, author
  email, SPDX-format license, removed deprecated MIT classifier.
- README primary install instruction is now `pip install osr-metrics`;
  editable install moved to a Development subsection.

### Removed
- Tracked build artifacts (`*.egg-info/`, `.pytest_cache/`) — they were
  always supposed to be ignored.
```

- [ ] **Step 3: Reinstall so metadata picks up the new version**

Run: `pip install -e .`
Expected: `Successfully installed osr-metrics-0.1.2`.

- [ ] **Step 4: Confirm version test passes against the bumped version**

Run: `pytest tests/test_version.py -v`
Expected: all 3 tests PASS, and `osr_metrics.__version__` is now `"0.1.2"`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.1.2"
```

---

## Task 6: Create `ci.yml` Workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: pytest (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install package and dev deps
        run: pip install -e .[dev]

      - name: Run tests
        run: pytest tests/ -v

  build:
    name: build sdist + wheel
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install build tooling
        run: pip install --upgrade build twine

      - name: Build distributions
        run: python -m build

      - name: Check distributions
        run: twine check dist/*

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: dist-${{ github.sha }}
          path: dist/
          retention-days: 14
```

- [ ] **Step 2: Lint the YAML locally**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml').read()); print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add pytest matrix + build + twine check workflow"
```

---

## Task 7: Create `release.yml` Workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  id-token: write   # required for Trusted Publishing OIDC
  contents: read

jobs:
  verify-and-build:
    name: verify tag and build
    runs-on: ubuntu-latest
    outputs:
      is-stable: ${{ steps.classify.outputs.is-stable }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Verify tag matches pyproject.toml version
        run: |
          set -euo pipefail
          tag="${GITHUB_REF_NAME#v}"
          declared=$(python -c "import tomllib; print(tomllib.loads(open('pyproject.toml').read())['project']['version'])")
          echo "Tag version:        $tag"
          echo "pyproject version:  $declared"
          if [ "$tag" != "$declared" ]; then
            echo "::error::Tag $GITHUB_REF_NAME does not match pyproject version $declared"
            exit 1
          fi

      - name: Classify tag (stable vs pre-release)
        id: classify
        run: |
          set -euo pipefail
          if [[ "${GITHUB_REF_NAME}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "is-stable=true" >> "$GITHUB_OUTPUT"
            echo "Stable release tag detected → will publish to PyPI"
          else
            echo "is-stable=false" >> "$GITHUB_OUTPUT"
            echo "Pre-release tag detected → will publish to TestPyPI"
          fi

      - name: Install package and dev deps
        run: pip install -e .[dev]

      - name: Run full test suite on tagged commit
        run: pytest tests/ -v

      - name: Install build tooling
        run: pip install --upgrade build twine

      - name: Build distributions
        run: python -m build

      - name: Check distributions
        run: twine check dist/*

      - name: Upload artifact for publish jobs
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
          retention-days: 7

  publish-testpypi:
    name: publish to TestPyPI
    needs: verify-and-build
    if: needs.verify-and-build.outputs.is-stable == 'false'
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - name: Download dist artifact
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    name: publish to PyPI
    needs: verify-and-build
    if: needs.verify-and-build.outputs.is-stable == 'true'
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - name: Download dist artifact
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 2: Lint the YAML locally**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml').read()); print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add tag-driven release workflow with trusted publishing"
```

---

## Task 8: Update README with Badges and Install Instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a badge row at the top of `README.md`**

Insert immediately after the `# osr-metrics` heading and before the existing description:

```markdown
[![PyPI version](https://img.shields.io/pypi/v/osr-metrics.svg)](https://pypi.org/project/osr-metrics/)
[![Python versions](https://img.shields.io/pypi/pyversions/osr-metrics.svg)](https://pypi.org/project/osr-metrics/)
[![License: MIT](https://img.shields.io/pypi/l/osr-metrics.svg)](https://github.com/hxtruong6/osr-metrics/blob/main/LICENSE)
[![CI](https://github.com/hxtruong6/osr-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/hxtruong6/osr-metrics/actions/workflows/ci.yml)
```

- [ ] **Step 2: Replace the `## Install` section**

Find:

```markdown
## Install

```bash
pip install -e .
# or, with dev tools:
pip install -e .[dev]
```

Requires Python 3.10+, `numpy`, `scikit-learn`, `scipy`.
```

Replace with:

```markdown
## Install

```bash
pip install osr-metrics
```

Requires Python 3.10+, `numpy`, `scikit-learn`, `scipy`.

### Development install

```bash
git clone https://github.com/hxtruong6/osr-metrics.git
cd osr-metrics
pip install -e .[dev]
```
```

- [ ] **Step 3: Verify rendering**

Run:
```bash
rm -rf dist/ && python -m build && twine check dist/*
```
Expected: both files `PASSED`. Then `rm -rf dist/`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add PyPI badges and pip install instructions"
```

---

## Task 9: Create `docs/RELEASING.md`

**Files:**
- Create: `docs/RELEASING.md`

- [ ] **Step 1: Create the file**

Create `docs/RELEASING.md`:

```markdown
# Releasing

`osr-metrics` releases are tag-driven. Bump the version, push a tag, the
GitHub Actions `release.yml` workflow does the rest.

## Versioning

Semantic versioning (MAJOR.MINOR.PATCH). The version lives in
`pyproject.toml`; `osr_metrics.__version__` reads it from installed
package metadata via `importlib.metadata`, so the two cannot drift after
install.

## Pre-release dry run (TestPyPI)

Always shake down a release on TestPyPI first. A broken release on real
PyPI burns the version number permanently.

1. Update `pyproject.toml`: `version = "X.Y.Z"`.
2. Update `CHANGELOG.md`: move `[Unreleased]` content into `[X.Y.Z] — YYYY-MM-DD`.
3. Commit:
   ```bash
   git commit -am "chore: bump version to X.Y.Z"
   ```
4. Push a pre-release tag:
   ```bash
   git tag vX.Y.Zrc1
   git push origin main vX.Y.Zrc1
   ```
5. Wait for the `Release` workflow to finish. It publishes to TestPyPI.
6. Verify in a clean venv:
   ```bash
   python -m venv /tmp/v && /tmp/v/bin/pip install \
     -i https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     osr-metrics==X.Y.Zrc1
   /tmp/v/bin/python -c "from osr_metrics import auroc, __version__; print(__version__)"
   ```
7. Visit `https://test.pypi.org/project/osr-metrics/X.Y.Zrc1/` and check
   that the README renders and sidebar links point at the right URLs.

## Stable release (PyPI)

1. Push the stable tag:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
2. The `Release` workflow runs and pauses for manual approval on the
   `pypi` GitHub Environment. Approve it.
3. Verify on PyPI:
   ```bash
   python -m venv /tmp/v && /tmp/v/bin/pip install osr-metrics==X.Y.Z
   /tmp/v/bin/python -c "from osr_metrics import __version__; print(__version__)"
   ```

## Tag patterns

The release workflow classifies tags by regex:

| Tag pattern | Index |
|---|---|
| `^v[0-9]+\.[0-9]+\.[0-9]+$` (e.g. `v0.1.2`) | PyPI |
| anything else starting with `v` (e.g. `v0.1.2rc1`, `v0.1.2.dev1`) | TestPyPI |

If the tag does not match the version in `pyproject.toml`, the workflow
fails before building anything.

## Recovering from a botched release

PyPI does not allow re-uploading the same version. If a stable release is
broken:
- Yank the broken version on PyPI (`pypi.org` → project → Manage → Yank).
- Bump to the next patch version, fix the issue, release again.

You can re-use the same TestPyPI version only by deleting the file there
first; in practice it's easier to just bump the rc number (`rc1` → `rc2`).
```

- [ ] **Step 2: Commit**

```bash
git add docs/RELEASING.md
git commit -m "docs: add release process checklist"
```

---

## Task 10: Push Branch and Verify CI Runs Green

This is the first end-to-end check that the workflow files are valid as far as GitHub is concerned. We push to `main` (or a PR branch) and watch `ci.yml` run.

**Files:** none modified.

- [ ] **Step 1: Push to GitHub**

Run: `git push origin main`
Expected: push succeeds.

- [ ] **Step 2: Watch the CI run**

Open `https://github.com/hxtruong6/osr-metrics/actions` (or run `gh run watch` if `gh` is installed).

Expected outcome:
- `test` job runs 3 times in matrix (Python 3.10, 3.11, 3.12) — all green.
- `build` job runs after `test` — green.
- The `dist-<sha>` artifact is downloadable from the run page.

- [ ] **Step 3: If anything is red, fix and re-push**

Common failures and fixes:
- `pip install -e .[dev]` fails → check `pyproject.toml` syntax.
- `pytest` failure on 3.10 only → likely a syntax/import that requires 3.11+; bump the minimum or fix.
- `twine check` fails → README has a markup PyPI rejects; fix the markup.

Re-push fixes as normal commits. Do **not** proceed to Task 11 until CI is green on `main`.

---

## Task 11: First Release Dry Run (TestPyPI via `v0.1.2rc1`)

The release workflow's tag/version verification step uses strict string equality. The `pyproject.toml` version must equal the tag (without the leading `v`) exactly. So for a pre-release, bump `pyproject.toml` to the rc version *first*, then tag.

**Files:**
- Modify: `pyproject.toml` (version bump, twice — to `0.1.2rc1`, then back to `0.1.2`).

- [ ] **Step 1: Bump `pyproject.toml` to the rc version**

Change `version = "0.1.2"` to `version = "0.1.2rc1"`.

- [ ] **Step 2: Reinstall and confirm metadata picks it up**

Run:
```bash
pip install -e . && python -c "import osr_metrics; print(osr_metrics.__version__)"
```
Expected: prints `0.1.2rc1`.

- [ ] **Step 3: Commit and push the rc bump, then tag**

Run:
```bash
git commit -am "chore: bump to 0.1.2rc1 for TestPyPI dry run"
git push origin main
git tag v0.1.2rc1
git push origin v0.1.2rc1
```

- [ ] **Step 4: Watch the release workflow**

In the Actions tab, the `Release` workflow runs.

Expected:
- `verify-and-build` job:
  - `Verify tag matches pyproject.toml version` step prints `Tag version: 0.1.2rc1` and `pyproject version: 0.1.2rc1`, succeeds.
  - `Classify tag` step prints `Pre-release tag detected → will publish to TestPyPI`.
  - Tests pass on Python 3.11.
  - Build succeeds, `twine check` passes.
- `publish-testpypi`: green (publishes via OIDC, no token).
- `publish-pypi`: skipped.

- [ ] **Step 5: Verify successful TestPyPI publication**

Expected workflow outcome:
- `verify-and-build`: green.
- `publish-testpypi`: green (`is-stable == false`).
- `publish-pypi`: skipped (`is-stable == false`).

Visit `https://test.pypi.org/project/osr-metrics/0.1.2rc1/`:
- README renders.
- Sidebar shows Homepage, Repository, Issues, Changelog links.
- Files section lists both the `.tar.gz` and `.whl`.

- [ ] **Step 6: Install from TestPyPI in a clean venv**

Run:
```bash
python -m venv /tmp/osr-test
/tmp/osr-test/bin/pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  osr-metrics==0.1.2rc1
/tmp/osr-test/bin/python -c "from osr_metrics import auroc, compute_aoscr, delong_test, __version__; print(__version__)"
rm -rf /tmp/osr-test
```
Expected: prints `0.1.2rc1`.

(The `--extra-index-url` is needed because TestPyPI doesn't host
numpy/scipy/scikit-learn dependencies.)

- [ ] **Step 7: Bump version back to clean stable for the real release**

Change `version = "0.1.2rc1"` back to `version = "0.1.2"` in `pyproject.toml`. Then:

```bash
git commit -am "chore: bump to 0.1.2 for stable release"
git push origin main
```

---

## Task 12: First Stable Release (PyPI via `v0.1.2`)

**Files:** none modified.

- [ ] **Step 1: Push the stable tag**

Run:
```bash
git tag v0.1.2
git push origin v0.1.2
```

- [ ] **Step 2: Watch the workflow**

Expected:
- `verify-and-build`: green.
- `publish-testpypi`: skipped (`is-stable == true`).
- `publish-pypi`: pauses on the `pypi` environment, awaiting approval.

- [ ] **Step 3: Approve the deployment**

In the workflow run page, click the "Review deployments" button, check the `pypi` environment, click "Approve and deploy".

- [ ] **Step 4: Verify successful PyPI publication**

`publish-pypi` should turn green.

Visit `https://pypi.org/project/osr-metrics/0.1.2/`:
- README renders.
- Sidebar links resolve.
- Both files listed.

- [ ] **Step 5: Install from PyPI in a clean venv**

Run:
```bash
python -m venv /tmp/osr-pypi
/tmp/osr-pypi/bin/pip install osr-metrics==0.1.2
/tmp/osr-pypi/bin/python -c "from osr_metrics import auroc, __version__; print(__version__)"
rm -rf /tmp/osr-pypi
```
Expected: prints `0.1.2`.

- [ ] **Step 6: Verify all acceptance criteria from the spec**

Run through the design doc's §6 acceptance criteria:
1. ✅ `ci.yml` green on push.
2. ✅ `v0.1.2rc1` published to TestPyPI; install works.
3. ✅ `v0.1.2` published to PyPI; install works.
4. ✅ No tokens stored (check `Settings → Secrets and variables → Actions` — should be empty).
5. ✅ `osr_metrics.__version__ == "0.1.2"` after install.
6. ✅ Working tree clean: `git status` shows nothing tracked from `*.egg-info/`, `dist/`, `build/`, `__pycache__/`, `.pytest_cache/`.

If all six pass, the pipeline is shipped.

---

## Done

After Task 12, future releases follow `docs/RELEASING.md`:
1. Bump `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Tag `vX.Y.ZrcN` → TestPyPI dry run → verify.
4. Tag `vX.Y.Z` → approve in `pypi` environment → live on PyPI.
