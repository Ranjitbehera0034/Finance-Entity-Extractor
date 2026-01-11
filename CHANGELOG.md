# Changelog

All notable changes to FinEE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- (Next features go here)

---

## [1.1.0] - 2026-01-12
### Added
- **Complete Data Pipeline** (`scripts/data_pipeline/`)
  - `step1_unify.py`: Unifies MBOX, JSON, CSV, XML sources
  - `step2_filter.py`: Removes OTPs, spam, marketing messages
  - `step3_baseline.py`: Tests regex extractor accuracy
  - `step4_label.py`: Creates labeled training data with ground truth

- **Synthetic Data Generator**
  - `generate_synthetic.py`: Production-grade grammar-based generator
    - 100K+ realistic Indian bank transactions
    - All major banks (HDFC, ICICI, SBI, Axis, Kotak, PNB, BOB, etc.)
    - Brokerages (Zerodha, Groww, Upstox, Angel One, 5Paisa, etc.)
    - E-commerce, food, travel, utilities, entertainment categories
  - `generate_advanced.py`: Advanced features
    - Markov Chain for realistic message flow
    - Real data calibration from actual samples
    - Multilingual support (Hindi, Tamil, Telugu, Bengali, Kannada)
    - Data augmentation and edge case oversampling

- **LLM Fine-tuning Pipeline** (`scripts/finetune.py`)
  - Supports MLX (Apple Silicon) and PyTorch backends
  - LoRA fine-tuning with automatic data preparation
  - Model fusion and evaluation utilities

### Performance
- Trained on 152,519 records (2,419 real + 100K synthetic + 50K multilingual)
- Val loss: 2.42 → 0.46 (81% reduction)
- 100% JSON parsing accuracy on test cases
- Multilingual extraction working (Hindi, Tamil, Telugu, Bengali, Kannada)
- Fine-tuned model: 7.6GB (Phi-3-mini + LoRA fused)

### Models
- Fine-tuned model: `finetuned-v1/` on Hugging Face
- LoRA adapters: `lora-adapters/` on Hugging Face

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
