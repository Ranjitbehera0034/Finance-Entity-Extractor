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

**Extract structured financial data from Indian banking messages.**
<br>
*94.5% field accuracy. <1ms latency. Zero setup.*

</div>

---

## ⚡ Install & Run in 10 Seconds

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

**No model download. No API keys. Works offline.**

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
  "vpa": "swiggy@ybl",        // string - Raw VPA
  "confidence": 0.95,         // float - 0.0 to 1.0
  "confidence_level": "HIGH"  // "LOW" | "MEDIUM" | "HIGH"
}
```

---

## 🔬 Verify Accuracy Yourself

Don't trust "99% accuracy" claims. **Run the benchmark:**

```bash
# Clone and test
git clone https://github.com/Ranjitbehera0034/Finance-Entity-Extractor.git
cd Finance-Entity-Extractor
pip install finee

# Run benchmark
python benchmark.py --all
```

**Test on YOUR data:**
```bash
python benchmark.py --file your_transactions.jsonl
```

---

## 💀 Torture Test (Edge Cases)

Real bank SMS is messy. Here's how FinEE handles the chaos:

| Edge Case | Input | Result |
|-----------|-------|--------|
| **Missing spaces** | `Rs.500.00debited from A/c1234` | ✅ amount=500.0 |
| **Weird formatting** | `Rs 2,500/-debited dt:28/12/25` | ✅ amount=2500.0 |
| **Mixed case** | `RS. 1500 DEBITED from ACCT` | ✅ amount=1500.0, type=debit |
| **Unicode symbols** | `₹2,500 debited from •••• 3545` | ✅ amount=2500.0 |
| **Multiple amounts** | `Rs.500 debited. Bal: Rs.15,000` | ✅ amount=500.0 (first) |
| **Truncated SMS** | `Rs.2500 debited from A/c...3545 to swi...` | ✅ amount=2500.0 |
| **Extra noise** | `ALERT! Dear Customer, Rs.500 debited... Ignore if done by you.` | ✅ amount=500.0 |

**Run torture tests:**
```bash
python benchmark.py --torture
```

---

## 🏦 Supported Banks

| Bank | Debit | Credit | UPI | NEFT/IMPS |
|------|:-----:|:------:|:---:|:---------:|
| HDFC | ✅ | ✅ | ✅ | ✅ |
| ICICI | ✅ | ✅ | ✅ | ✅ |
| SBI | ✅ | ✅ | ✅ | ✅ |
| Axis | ✅ | ✅ | ✅ | ✅ |
| Kotak | ✅ | ✅ | ✅ | ✅ |

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
│ TIER 1: Regex Engine (50+ battle-tested patterns)          │
│ Extract: amount, date, reference, account, vpa, type       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: Rule-Based Mapping (200+ VPA → merchant)           │
│ Map: vpa → merchant, merchant → category                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: LLM (Optional, for edge cases)                     │
│ Targeted prompts for: merchant, category only              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
ExtractionResult (Guaranteed Schema)
```

---

## 📊 Benchmark Results

| Metric | Value |
|--------|-------|
| **Field Accuracy** | 94.5% |
| **Latency (Regex)** | <1ms |
| **Latency (LLM)** | ~50ms |
| **Throughput** | 50,000+ msg/sec |
| **Banks Tested** | 5 (HDFC, ICICI, SBI, Axis, Kotak) |

---

## 💻 CLI Usage

```bash
# Extract from text
finee extract "Rs.500 debited from A/c 1234"

# Show version
finee --version

# Check available backends
finee backends
```

---

## 📁 Repository Structure

```
Finance-Entity-Extractor/
├── src/finee/              # Core package (16 modules)
│   ├── extractor.py        # Pipeline orchestrator
│   ├── regex_engine.py     # 50+ regex patterns
│   ├── merchants.py        # 200+ VPA mappings
│   └── backends/           # MLX, PyTorch, GGUF
├── tests/                  # 88 unit tests
├── examples/               # Colab notebook
├── experiments/            # Research notebooks
├── benchmark.py            # ⭐ Verify accuracy yourself
├── pyproject.toml
└── README.md
```

---

## 🤝 Contributing

```bash
git clone https://github.com/Ranjitbehera0034/Finance-Entity-Extractor.git
cd Finance-Entity-Extractor
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**Made with ❤️ by Ranjit Behera**

[PyPI](https://pypi.org/project/finee/) · [GitHub](https://github.com/Ranjitbehera0034/Finance-Entity-Extractor) · [Hugging Face](https://huggingface.co/Ranjit0034/finance-entity-extractor)

</div>
