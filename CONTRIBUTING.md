# Contributing to FinEE

Thank you for your interest in contributing! Here's how to get started.

## 🌿 Branching Strategy

We use **Git Flow** for professional development:

```
main          ────●────●────●────────────●─────→  (stable releases only)
                  │    │    │            │
                  │    │    └──tag:v1.0.3│
                  │    │                 │
develop       ────●────●────●────●───────●─────→  (integration branch)
                       │    │    │
feature/xyz   ─────────●────●────┘               (feature branches)
```

### Branches

| Branch | Purpose | Merge To |
|--------|---------|----------|
| `main` | Stable releases only | - |
| `develop` | Integration & testing | `main` (via PR) |
| `feature/*` | New features | `develop` (via PR) |
| `fix/*` | Bug fixes | `develop` (via PR) |
| `hotfix/*` | Urgent production fixes | `main` + `develop` |

### Workflow

1. **New Feature**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/my-feature
   # ... make changes ...
   git push -u origin feature/my-feature
   # Create PR to develop
   ```

2. **Bug Fix**:
   ```bash
   git checkout develop
   git checkout -b fix/issue-123
   # ... fix bug ...
   git push -u origin fix/issue-123
   # Create PR to develop
   ```

3. **Release**:
   ```bash
   git checkout main
   git merge develop
   git tag -a v1.x.x -m "Release v1.x.x"
   git push origin main --tags
   ```

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_regex_engine.py -v

# Run with coverage
pytest tests/ --cov=finee --cov-report=html
```

## 📝 Code Style

- **Black** for formatting (line length: 100)
- **Ruff** for linting
- **Type hints** for all public functions

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/finee/
```

## 🚀 Publishing

Only maintainers can publish to PyPI:

```bash
# Bump version in pyproject.toml
# Build
python -m build

# Upload
twine upload dist/*
```

## 📋 Commit Messages

Use conventional commits:

```
feat: add support for Lakhs notation
fix: handle Unicode ₹ symbol in regex
docs: update README with torture tests
test: add edge case tests for truncated SMS
chore: move notebooks to experiments/
```

## 🙏 Thank You!

Every contribution helps make FinEE better for the Indian fintech community.
