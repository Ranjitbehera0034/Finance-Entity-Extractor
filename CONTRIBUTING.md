# Contributing to FinEE

Thank you for your interest in contributing! Here's how to get started.

## 🚨 Golden Rules

1. **NEVER push directly to `main`** - All changes go through PRs
2. **All features branch from `develop`** - Not from main
3. **Tests MUST pass before merging** - No exceptions
4. **Update CHANGELOG.md** - Every PR should update it

## 🌿 Branching Strategy (Git Flow)

```
main          ──●────────────────●────────────●──→  (releases only)
                │                │            │
              v1.0.3          v1.0.4        v1.1.0
                │                │            │
develop       ──●────●────●──────●───●────●───●──→  (integration)
                     │    │          │    │
feature/a     ───────●────┘          │    │
feature/b     ───────────────────────●────┘
```

### Branch Types

| Branch | Source | Merges To | Naming |
|--------|--------|-----------|--------|
| `main` | - | - | Protected, releases only |
| `develop` | `main` | `main` | Integration branch |
| `feature/*` | `develop` | `develop` | `feature/add-kotak-support` |
| `fix/*` | `develop` | `develop` | `fix/unicode-amount-parsing` |
| `hotfix/*` | `main` | `main` + `develop` | `hotfix/critical-regex-bug` |

## 🔄 Development Workflow

### 1. Start a New Feature

```bash
# Always start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/my-awesome-feature

# Make changes...
# Write tests...
# Update CHANGELOG.md under [Unreleased]

# Commit with conventional messages
git commit -m "feat: add support for Paytm VPA patterns"

# Push and create PR
git push -u origin feature/my-awesome-feature
# Create PR: feature/my-awesome-feature → develop
```

### 2. Fix a Bug

```bash
git checkout develop
git pull origin develop
git checkout -b fix/issue-123-unicode-error

# Fix the bug...
# Add test to prevent regression...

git commit -m "fix: handle ₹ symbol in amount parsing"
git push -u origin fix/issue-123-unicode-error
# Create PR: fix/issue-123-unicode-error → develop
```

### 3. Create a Release

**Use the release script:**

```bash
# Preview what will happen
python scripts/release.py 1.0.4 --dry-run

# Execute release
python scripts/release.py 1.0.4
```

The script will:
1. ✅ Verify you're on `develop`
2. ✅ Run all tests
3. ✅ Run benchmark suite
4. ✅ Update version in `pyproject.toml`
5. ✅ Update `CHANGELOG.md`
6. ✅ Merge to `main`
7. ✅ Create git tag `v1.0.4`
8. ✅ Build and upload to PyPI
9. ✅ Return to `develop`

## 🧪 Pre-Merge Checklist

Before your PR can be merged:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Benchmark runs: `python benchmark.py --all`
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Code formatted: `black src/ tests/`
- [ ] Linting passes: `ruff check src/ tests/`
- [ ] New features have tests
- [ ] Documentation updated if needed

## 📋 Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding tests |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `chore` | Maintenance tasks |

### Examples

```
feat(regex): add Lakhs notation support
fix(parser): handle missing spaces in SMS
docs: update README with torture tests
test: add edge cases for Unicode symbols
chore: move notebooks to experiments/
```

## 📝 Changelog Guidelines

Update `CHANGELOG.md` in every PR under `[Unreleased]`:

```markdown
## [Unreleased]
### Added
- New feature you added

### Changed
- Behavior you modified

### Fixed
- Bug you fixed
```

**Categories:**
- **Added** - New features
- **Changed** - Changes to existing features
- **Deprecated** - Features to be removed
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security fixes

## 🛡️ Protected Branches

| Branch | Direct Push | Force Push | PR Required |
|--------|-------------|------------|-------------|
| `main` | ❌ | ❌ | ✅ (from develop only) |
| `develop` | ❌ | ❌ | ✅ (from feature/fix) |

## 🙏 Thank You!

Every contribution makes FinEE better for the Indian fintech community!
