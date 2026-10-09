# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

manipy: manifold alignment, dimensionality reduction and hyperspectral (HSI)
workflows in JAX. It sits on top of
[kernellib](https://github.com/jejjohnson/kernellib) (graphs, Laplacians,
eigenmaps, kernel PCA, kernel methods) and
[gaussx](https://github.com/jejjohnson/gaussx) (structured operators, solvers,
`eigh_generalized`, `SparseOperator`). It owns manifold alignment,
non-spectral-graph embeddings (Isomap, the LLE family, diffusion maps),
out-of-sample extension, HSI workflows, embedding-quality and classification
metrics, dataset loaders, and scikit-learn adapters. Built with Python 3.12+,
uv, pytest, mystmd, and MkDocs.

The PyPI distribution is **`manipy-jax`** (`manipy` is taken by an unrelated
project); the import package is `manipy`.

The plan lives in kernellib's published roadmap, `docs/roadmap/roadmap-manipy.md`
(<https://github.com/jejjohnson/kernellib/blob/main/docs/roadmap/roadmap-manipy.md>):
scope, package layout, the manifold alignment specification, and the phases
M0 to M5. Read it before adding a module. Work is tracked as GitHub issues.

The 2018 `manilearn/` code is preserved on the `legacy` branch and the
`v0.0.0-legacy` tag. It is a feature list and a numerical reference, never
code to copy: several of its modules never ran.

### Boundaries

- **manipy imports kernellib and gaussx; they never import manipy.** Anything that is a graph, a Laplacian, a graph kernel, Laplacian / Schrödinger eigenmaps, LPP / SEP or kernel PCA lives in kernellib. manipy **uses** it and never re-exports it: users import `kernellib.LaplacianEigenmaps` directly, and the docs say so. If a needed primitive is missing, add it to kernellib or gaussx, not here.
- **manipy never imports `numpyro` or pyrox.** `import manipy` must load neither (`tests/test_imports.py` enforces it).
- **The core never imports scikit-learn.** scikit-learn appears only in the adapter subpackage `manipy.sklearn` (extra `manipy-jax[sklearn]`), which wraps core objects in scikit-learn's estimator API and is never imported from the core. `import manipy` must not load `sklearn`.
- Optional extras (`neighbors`, `data`) are imported inside the function that needs them, only when chosen. `pooch`, `scipy.io` and pynndescent never load on `import manipy`.
- **kernellib and gaussx are pinned by git tag** in `[tool.uv.sources]` until they are on PyPI, with a matching `>=` floor in `dependencies`. kernellib pins its own gaussx; `[tool.uv] override-dependencies` keeps one gaussx for the whole resolve. Bump the kernellib tag, the gaussx tag and the override together.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Fast tier (the default): uv run pytest -n auto
make test-slow            # Slow tier: -m "slow and not integration"
make test-integration     # Integration tier: -m integration
make test-all             # Every tier: -m ""
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/manipy scripts
make precommit            # Run pre-commit on all files
make docs-serve           # Build the docs, then serve them locally
```

### Running a single test

```bash
uv run pytest tests/test_public_api.py::test_all_is_sorted_and_unique -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -n auto                         # Fast tests + doctests (add `-m ""` for all tiers)
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/manipy/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/manipy scripts  # Typecheck
```

**Critical**: Always lint/format with `.` (repo root), not `src/manipy/`. CI runs `ruff check .`, which includes `tests/`, `scripts/`, and the code cells of any `docs/notebooks/*.ipynb`.

`--doctest-modules` is in `addopts` and `src/manipy` is a `testpath`, so every `Examples:` block in a docstring is executed on each run. When you change behaviour, update the examples, and verify the expected output against what the code actually prints.

### Building the docs

```bash
make docs        # what CI runs: builds both halves and verifies every link
make docs-api    # API reference only — fast, and needs no Node
```

See the [Documentation](#documentation) section below; `mkdocs build` alone
covers only the API half.

## Architecture

### Package structure

All implementation lives in `src/manipy/`. The public API is flat and re-exported through `src/manipy/__init__.py`; `__all__` there is the contract, and `tests/test_public_api.py` enforces it. Every new top-level module must be added to `SUBMODULES` in that test and given an API doc page. Subpackages are private (`_name`) except `sklearn`.

The planned layout (roadmap §3). Directories are created by the phase that needs them, not ahead of time; at M0 only `__init__.py` existed:

| Path | Contents | Phase |
|---|---|---|
| `__init__.py` | Flat public API; `__all__` is the contract | M0 |
| `_alignment/` | `_domains.py` (`Domain` container, joint block assembly), `_class_graphs.py` (same / different-label quadratic forms, SSMA rescaling), `_linear.py` (`ManifoldAlignment`: `"wang"`, `"ssma"`, `"sema"`), `_kernel.py` (`KernelManifoldAlignment`, KEMA) | M1, M4 |
| `_embeddings/` | `_isomap.py` (Isomap, landmark Isomap), `_lle.py` (LLE, modified LLE, Hessian LLE, LTSA), `_diffusion_maps.py` (anisotropic α), `_tsne.py` | M3, M4 |
| `_out_of_sample.py` | Nyström extension for LE / SE / diffusion maps | M3 |
| `_hsi/` | `_image.py` (`image_to_array` / `array_to_image`, `pixel_coordinates`, `pixel_graph`), `_potentials.py` (image-level Schrödinger potentials on top of kernellib), `_splits.py` (`stratified_split`) | M2 |
| `_metrics/` | `_quality.py` (trustworthiness, continuity, LCMC, `knn_preservation`), `_classification.py` (overall / average / per-class accuracy, Cohen's kappa) | M2 |
| `_datasets/` | `_synthetic.py` (`swiss_roll`, `s_curve`, `severed_sphere`; JAX, keyed), `_hsi.py` (Indian Pines, Pavia University, Salinas; pooch download with checksums) | M2 |
| `sklearn/` | scikit-learn adapters, extra `manipy-jax[sklearn]` | with each estimator |

Dependency direction is one-way: foundations (`_datasets`, `_metrics`, `_hsi`) → methods (`_alignment`, `_embeddings`, `_out_of_sample`) → adapters (`sklearn`). Nothing in a lower layer imports a higher one.

### Key directories

| Path | Purpose |
|------|---------|
| `src/manipy/` | Main package source code |
| `tests/` | Test suite |
| `docs/guide/` | Hand-written guide pages (created when the first one is written) |
| `docs/roadmap/` | Pointer to kernellib's roadmap; the roadmap itself is published from kernellib |
| `docs/api/` | mkdocstrings API reference, one page per module |
| `docs/notebooks/` | Executed example notebooks (outputs committed; created with the first one) |
| `scripts/` | Build tooling, incl. `build_docs.py` (the two-tool docs pipeline) |

## Test Speed Tiers

- Unmarked (**fast**, the default selection): unit tests and doctests, < ~1 s each.
- `@pytest.mark.slow`: individually expensive tests (> ~1 s — heavy numerics, `jit`+`grad`+`vmap` sweeps, convergence checks, subprocess import checks).
- `@pytest.mark.integration`: end-to-end workflows — scikit-learn `check_estimator` sweeps, cross-library pipelines (kernellib / gaussx), dataset downloads, optional backends. Doctests under `manipy.sklearn.` are marked integration automatically by `tests/conftest.py`; slow doctests are listed in `_SLOW_DOCTESTS` there, since a doctest cannot take a decorator.

`addopts` selects `-m "not slow and not integration"`, so plain `uv run pytest` runs the fast tier. The last `-m` wins: `uv run pytest -m slow`, `-m integration`, or `-m ""` for everything. Mark a new test `slow` if it takes more than about a second. CI runs the three tiers as parallel jobs and gates coverage on their union (`fail_under` in `pyproject.toml`), so no single tier has to reach it; locally, `make test-cov` runs every tier with coverage. The gate is 90, as in kernellib.

## Tests That Assert On Random Draws

- **If the randomness is incidental** — the test checks a correctness property and any draw would do — pin the key (`jax.random.key(0)`). Deterministic makes the tolerance mean something.
- **If the test is genuinely about sampling behaviour**, bound the estimator by its own sampling distribution rather than a fixed `atol`, and say in a comment where the bound came from.

## Documentation

The docs are built by **two tools** and deployed as one site — see
`docs/README.md` for the full rationale.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, notebooks, roadmap pointer | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/`, `docs/roadmap/` | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` | `/reference/` |

```bash
make docs          # build both halves, assemble into public/, verify links
make docs-api      # API reference only (fast; no Node needed)
make docs-serve    # build, then serve the assembled site at :8000
```

`scripts/build_docs.py` orchestrates this. It serves the freshly built
`site/` on port 8910 so mystmd can read the `objects.inv`, rewrites the
resulting localhost URLs to `/reference/`, repairs anchors broken by the
mystmd `$`-expansion bug, and then verifies that every internal link in the
assembled site resolves. Its pure functions are covered by
`tests/test_build_docs.py`.

**mystmd is a Node CLI**: `npm install -g mystmd`. It is not a uv dependency.

### Writing prose

Prose pages are **MyST Markdown**, not MkDocs-Material Markdown. Use
`:::{note}` / `:::{tab-set}` / `:::{dropdown}` directives, not `!!!` / `===`
/ `???` blocks.

Cross-reference the API with the `xref:` protocol and the **top-level**
exported name:

```markdown
[`ManifoldAlignment`](xref:api#manipy.ManifoldAlignment)           <!-- correct -->
[`ManifoldAlignment`](xref:api#manipy._alignment.ManifoldAlignment) <!-- avoid: see docs/README.md -->
```

A target missing from the inventory fails `myst build --strict`. Every module
gets an API page under `docs/api/` and an entry in `mkdocs.yml`'s `nav` and
in `docs/api/index.md`.

Every API page and notebook states its formulation (the maths matched to the
code), the pseudocode of the algorithm, and verified references, cited with
MyST citations from a `.bib` file under `docs/bib/` (add it to `bibliography`
in `docs/myst.yml`).

### URLs are flat

mystmd derives a page's URL from its **basename**, so `guide/architecture.md`
would be served at `/architecture/`, not `/guide/architecture/`. Keep basenames
unique across `docs/guide/`, `docs/notebooks/` and `docs/roadmap/`. Frontmatter
`slug:` is ignored. Underscores become hyphens: `notebooks/hsi_workflow.ipynb`
is served at `/hsi-workflow/`, which matters when an API page links into the
prose half with a relative URL such as `../../hsi-workflow/`.

## Documentation Examples

Example notebooks live in `docs/notebooks/` as executed `.ipynb` files with
their outputs committed; mystmd renders them without re-executing. Author
them in jupytext percent format, execute, then delete the `.py`. Notebooks
may use docs-group tools (matplotlib, scikit-learn); nothing under `src/`
may import them.

## Coding Conventions

- Estimators are `equinox.Module` subclasses (immutable, PyTree-compatible): configuration in the constructor, `fit` returns the fitted module, and fitted fields are `None` before `fit`. Helpers are pure functions.
- Use `jaxtyping` annotations for array shapes
- Use `einx` for **every operation on a dense array** (anything that is not a lineax operator), in `src/` and `tests/`:
  - contractions and transposes: `einx.dot("j i, j -> i", A, x)`, never `A.T @ x`, `jnp.einsum` or `jnp.transpose`;
  - reshapes: `einx.rearrange`, never `.reshape` or `jnp.reshape`;
  - axis reductions: `einx.mean("i [j] -> i", K)`, never `jnp.mean(K, axis=0)` (likewise `sum`, `max`, `std`, ...);
  - broadcasting against inserted axes: `einx.subtract("i j, j -> i j", K, col)`, never `K - col[None, :]` (likewise `einx.add`, `einx.multiply`, ...).

  Fine as they are: matvecs with no transpose (`L @ z`), full reductions (`jnp.sum(A)`), `jnp.eye` / `jnp.diag`, `jnp.concatenate` / `jnp.stack`, and lineax operator methods. Before committing, grep the diff for `axis=`, `[:, None]`, `[None, :]`, `.T` and `reshape`.
- Google-style docstrings with executable `Examples:` blocks
- Type hints on all public functions and methods
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Plans

Plans go in `.plans/` (gitignored, never committed). Track work via GitHub
issues. The published roadmap lives in kernellib's `docs/roadmap/`; manipy's
`docs/roadmap/` only points at it.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

**Automated reviewers (GitHub Copilot and ChatGPT Codex only).** Address their comments with code changes, without replying to them. Once a comment is addressed (or deliberately declined), resolve its thread and then **hide the bot's comment as resolved** with the `minimizeComment` mutation (`classifier: RESOLVED`). Their authors are `copilot-pull-request-reviewer` and `chatgpt-codex-connector`. Never hide human reviewers' comments. On stacked PRs (base is another PR's branch), Codex posts its review as a plain PR comment instead of review threads, so check both `reviewThreads` and `comments`. See `AGENTS.md` for the queries.

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
