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
