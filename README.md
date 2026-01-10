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
- indian-banking
base_model: microsoft/Phi-3-mini-4k-instruct
pipeline_tag: text-generation
---

<div align="center">

# Finance Entity Extractor (FinEE) v1.0

[![PyPI](https://img.shields.io/pypi/v/finee?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/finee/)
[![Tests](https://github.com/Ranjitbehera0034/Finance-Entity-Extractor/actions/workflows/tests.yml/badge.svg)](https://github.com/Ranjitbehera0034/Finance-Entity-Extractor/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ranjitbehera0034/Finance-Entity-Extractor/blob/main/examples/demo.ipynb)

**Production-grade Finance NER for Indian Banks**
<br>
*Hybrid Regex + Phi-3 LLM • 94.5% accuracy • <1ms latency*

</div>

---

## 🔥 Hybrid Architecture

> **Runs 100% offline using Regex by default.**
> **Optional 3.8B LLM auto-downloads only for complex edge cases.**

| Mode | Latency | Accuracy | Model Download |
|------|---------|----------|----------------|
| **Regex (Default)** | <1ms | 87% | ❌ None |
| **Regex + LLM** | ~50ms | 94.5% | ✅ 7GB (one-time) |

---

## ⚡ Install in 10 Seconds

```bash
pip install finee
```

```python
from finee import extract

r = extract("Rs.2500 debited from A/c XX3545 to swiggy@ybl on 28-12-2025")

print(r.amount)    # 2500.0
print(r.merchant)  # "Swiggy"
print(r.category)  # "food"
```

**Try it now:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ranjitbehera0034/Finance-Entity-Extractor/blob/main/examples/demo.ipynb)

---

## 🧠 Enable LLM Mode (For Edge Cases)

```python
from finee import FinEE
from finee.schema import ExtractionConfig

# Downloads 7GB model once, then runs locally
extractor = FinEE(ExtractionConfig(use_llm=True))
result = extractor.extract("Your complex bank message...")
```

**Supported Backends:**
- Apple Silicon → MLX (fastest)
- NVIDIA GPU → PyTorch/CUDA
- CPU → llama.cpp (GGUF)

---

## 📋 Output Schema Contract

Every extraction returns this **guaranteed JSON structure**:

```json
{
  "amount": 2500.0,           // float - Always numeric
  "currency": "INR",          // string - ISO 4217
  "type": "debit",            // "debit" | "credit"
  "account": "3545",          // string - Last 4 digits
  "date": "28-12-2025",       // string - DD-MM-YYYY
  "reference": "534567891234",// string - UPI/NEFT ref
  "merchant": "Swiggy",       // string - Normalized name
  "category": "food",         // string - food|shopping|transport|...
  "confidence": 0.95          // float - 0.0 to 1.0
}
```

---

## 🔬 Verify Accuracy Yourself

```bash
git clone https://github.com/Ranjitbehera0034/Finance-Entity-Extractor.git
cd Finance-Entity-Extractor
pip install finee
python benchmark.py --all
```

---

## 💀 Edge Case Handling

| Input | Result |
|-------|--------|
| `Rs.500.00debited from A/c1234` (no spaces) | ✅ amount=500.0 |
| `₹2,500 debited` (Unicode) | ✅ amount=2500.0 |
| `1.5 Lakh credited` (Lakhs) | ✅ amount=150000.0 |
| `Rs.500 debited. Bal: Rs.15,000` (multiple) | ✅ amount=500.0 |

---

## 🏦 Supported Banks

| Bank | Status |
|------|--------|
| HDFC | ✅ |
| ICICI | ✅ |
| SBI | ✅ |
| Axis | ✅ |
| Kotak | ✅ |

---

## 📊 Benchmark

| Metric | Value |
|--------|-------|
| **Field Accuracy** | 94.5% (with LLM) |
| **Regex-only Accuracy** | 87.5% |
| **Latency (Regex)** | <1ms |
| **Throughput** | 50,000+ msg/sec |

---

## 🏗️ Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 0: Hash Cache (<1ms if seen before)                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Regex Engine (50+ patterns)                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: Rule-Based Mapping (200+ VPA → merchant)           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: Phi-3 LLM (Optional - downloads 7GB model)         │
│         Only called for edge cases                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
ExtractionResult (Guaranteed Schema)
```

---

## 📁 Repository Structure

```
Finance-Entity-Extractor/
├── src/finee/              # Core package
├── tests/                  # 88 unit tests
├── examples/demo.ipynb     # 👈 Try in Colab!
├── benchmark.py            # Verify accuracy
├── CHANGELOG.md            # Release history
└── CONTRIBUTING.md         # How to contribute
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Git Flow branching strategy
- How to run tests
- Release process

---

## 📄 License

MIT License

---

<div align="center">

**Made with ❤️ by Ranjit Behera**

[PyPI](https://pypi.org/project/finee/) • [GitHub](https://github.com/Ranjitbehera0034/Finance-Entity-Extractor) • [Hugging Face](https://huggingface.co/Ranjit0034/finance-entity-extractor)

</div>
