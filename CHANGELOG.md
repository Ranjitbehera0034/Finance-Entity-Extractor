# Changelog

All notable changes to FinEE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- (Next features go here)

### Changed
- (Changes to existing features)

### Fixed
- (Bug fixes)

---

## [1.0.3] - 2025-01-11
### Added
- Lakhs notation support (e.g., "1.5 Lakh" → 150000)
- `benchmark.py` script for accuracy verification
- Torture test suite for edge cases
- `CONTRIBUTING.md` with Git Flow guidelines
- Professional branching: main, develop, feature/*

### Changed
- Moved Jupyter notebooks to `experiments/` folder
- Default `use_llm=False` for instant usage (no model download)
- Updated README with edge case examples

### Fixed
- Double-escaped regex patterns in amount extraction

---

## [1.0.2] - 2025-01-11
### Changed
- Default to regex-only mode (`use_llm=False`)
- Package works instantly without downloading 5GB model

### Fixed
- Package build configuration (hatch sources)

---

## [1.0.1] - 2025-01-11
### Fixed
- Package did not include source files (only 5KB)
- Fixed `pyproject.toml` build configuration

---

## [1.0.0] - 2025-01-11
### Added
- Initial PyPI release
- 5-tier additive extraction pipeline (Cache/Regex/Rules/LLM/Validate)
- Multi-backend support (MLX, PyTorch, GGUF)
- CLI with `finee` command
- 88 unit tests
- Colab demo notebook
- JSON schema contract documentation
- Support for 5 banks: HDFC, ICICI, SBI, Axis, Kotak

### Performance
- 94.5% field accuracy on multi-bank benchmark
- <1ms latency in regex-only mode
- 50,000+ messages/second throughput

---

## Links
- [PyPI](https://pypi.org/project/finee/)
- [GitHub](https://github.com/Ranjitbehera0034/Finance-Entity-Extractor)
- [Hugging Face](https://huggingface.co/Ranjit0034/finance-entity-extractor)
