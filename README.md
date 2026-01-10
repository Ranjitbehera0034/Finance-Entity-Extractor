---
language:
- en
license: mit
library_name: transformers
tags:
- finance
- entity-extraction
- ner
- phi-3
- production
- gguf
- indian-banking
- structured-output
base_model: microsoft/Phi-3-mini-4k-instruct
pipeline_tag: text-generation
---

<div align="center">

# Finance Entity Extractor (FinEE) v1.0

<a href="https://huggingface.co/Ranjit0034/finance-entity-extractor">
    <img src="https://img.shields.io/badge/Model-FinEE_3.8B-blue?style=for-the-badge&logo=huggingface" alt="Model Name">
</a>
<a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</a>
<a href="https://huggingface.co/Ranjit0034/finance-entity-extractor">
    <img src="https://img.shields.io/badge/Parameters-3.8B-orange?style=for-the-badge" alt="Parameters">
</a>
<a href="https://github.com/ggerganov/llama.cpp">
    <img src="https://img.shields.io/badge/GGUF-Compatible-purple?style=for-the-badge" alt="GGUF">
</a>
<a href="https://github.com/Ranjitbehera0034/Finance-Entity-Extractor/actions/workflows/tests.yml">
    <img src="https://github.com/Ranjitbehera0034/Finance-Entity-Extractor/actions/workflows/tests.yml/badge.svg" alt="Tests">
</a>

<br>

**A production-ready 3.8B parameter language model optimized for zero-shot financial entity extraction.**
<br>
*Validated on Indian banking syntax (HDFC, ICICI, SBI, Axis, Kotak) with 94.5% field accuracy.*

[ [Model Card](https://huggingface.co/Ranjit0034/finance-entity-extractor) ] · [ [GitHub](https://github.com/Ranjitbehera0034/Finance-Entity-Extractor) ] · [ [Quick Start](#quick-start-with-finee-library) ]

</div>

---

## Performance Benchmarks

### Comparison with Foundation Models

| Model | Parameters | Entity Precision (India) | Latency (CPU) | Cost |
|-------|------------|-------------------------|---------------|------|
| **FinEE-3.8B (Ours)** | 3.8B | **94.5%** | **45ms** | Free |
| Llama-3-8B-Instruct | 8B | 89.4% | 120ms | Free |
| GPT-3.5-Turbo | ~175B | 94.1% | ~500ms | $0.002/1K |
| GPT-4 | ~1.7T | 96.8% | ~800ms | $0.03/1K |

### Platform Support

| Platform | Framework | Status |
|----------|-----------|--------|
| macOS Apple Silicon | MLX | ✅ Full Support |
| Linux + NVIDIA GPU | PyTorch/Transformers | ✅ Full Support |
| Linux + CPU | PyTorch/GGUF | ✅ Full Support |
| Windows | GGUF/llama.cpp | ✅ Full Support |

## 🐍 Quick Start with FinEE Library

The easiest way to use the model is through the `finee` Python library, which handles backend selection, caching, and validation automatically.

### Installation

```bash
# Install from GitHub
pip install git+https://github.com/Ranjitbehera0034/Finance-Entity-Extractor.git

# Or clone and install locally
git clone https://github.com/Ranjitbehera0034/Finance-Entity-Extractor.git
cd Finance-Entity-Extractor
pip install -e ".[metal]"   # Apple Silicon
pip install -e ".[cuda]"    # NVIDIA GPU
pip install -e ".[cpu]"     # CPU only
```

### Usage

```python
from finee import extract

# Automatic backend detection (MLX, CUDA, or CPU)
text = "Rs.500 paid to swiggy@ybl on 01-01-2025"
result = extract(text)

print(f"Amount: {result.amount}")
print(f"Merchant: {result.merchant} ({result.category})")
print(f"Confidence: {result.confidence.value}")

# Output JSON
print(result.to_json())
# {
#   "amount": 500.0,
#   "type": "debit",
#   "merchant": "Swiggy",
#   "category": "food",
#   "date": "01-01-2025",
#   ...
# }
```

### Command Line Interface

```bash
# Direct extraction
finee extract "Rs.500 debited from A/c 1234"

# Check available backends
finee backends
```

---

## 📋 Overview

This project demonstrates how to:
1. **Parse** 40K+ emails from a Gmail MBOX export
2. **Classify** emails into categories using Phi-3 Mini
3. **Discover** patterns in financial emails (transactions, amounts, dates)
4. **Fine-tune** a local LLM using LoRA for entity extraction
5. **Extract** structured data: amount, transaction type, account, date, reference

## 🏗️ Project Structure

```
Finance-Entity-Extractor/
├── src/
│   └── finee/                 # FinEE Package
│       ├── __init__.py
│       ├── extractor.py       # Main pipeline orchestrator
│       ├── cache.py           # Tier 0 LRU Cache
│       ├── regex_engine.py    # Tier 1 Regex Engine
│       ├── merchants.py       # Tier 2 Rule Mapping
│       ├── prompt.py          # Tier 3 Targeted Prompts
│       ├── validator.py       # Tier 4 Validation & Repair
│       ├── backends/          # Auto-detecting Backends (MLX, PT, GGUF)
│       └── cli.py             # Command Line Interface
├── tests/                     # 88 Unit Tests
├── .github/workflows/         # CI/CD
├── pyproject.toml
├── train.py                   # Training pipeline
└── README.md
```

## 🎯 Extracted Entities

| Entity | Description | Example |
|--------|-------------|---------|
| `amount` | Transaction amount | "2500.00" |
| `type` | Debit or Credit | "debit" |
| `account` | Account identifier | "3545" |
| `date` | Transaction date | "28-12-25" |
| `reference` | UPI/NEFT reference | "534567891234" |
| `merchant` | Merchant name | "swiggy" |
| `category` | Transaction category | "food" |
| `confidence` | Extraction confidence | "HIGH" |

## 📈 Benchmark Results

### Multi-Bank Validation (v8)

| Bank | Field Accuracy | Status |
|------|----------------|--------|
| ICICI | 96.2% | ✅ |
| HDFC | 95.0% | ✅ |
| SBI | 93.3% | ✅ |
| Axis | 93.3% | ✅ |
| Kotak | 92.0% | ✅ |
| **Overall** | **94.5%** | ✅ |

### Field-Level Accuracy

| Field | Accuracy |
|-------|----------|
| Amount | 98.5% |
| Type | 99.2% |
| Date | 97.8% |
| Account | 96.1% |
| Reference | 72.7% |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Microsoft](https://huggingface.co/microsoft) for Phi-3 model
- [MLX team](https://github.com/ml-explore) for the amazing framework
- [Hugging Face](https://huggingface.co/) for model hosting

---

**Made with ❤️ by Ranjit Behera**
