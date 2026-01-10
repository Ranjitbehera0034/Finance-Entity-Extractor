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

<a href="https://pypi.org/project/finee/">
    <img src="https://img.shields.io/pypi/v/finee?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI">
</a>
<a href="https://github.com/Ranjitbehera0034/Finance-Entity-Extractor/actions/workflows/tests.yml">
    <img src="https://github.com/Ranjitbehera0034/Finance-Entity-Extractor/actions/workflows/tests.yml/badge.svg" alt="Tests">
</a>
<a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</a>
<a href="https://colab.research.google.com/github/Ranjitbehera0034/Finance-Entity-Extractor/blob/main/examples/demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

<br>

**Extract structured financial data from Indian banking messages in one command.**
<br>
*94.5% field accuracy across HDFC, ICICI, SBI, Axis, Kotak.*

</div>

---

## ⚡ One-Command Installation

```bash
pip install finee
```

That's it. No cloning, no setup.

---

## 🚀 30-Second Quick Start

```python
from finee import extract

# Parse any Indian bank message
result = extract("Rs.2500 debited from A/c XX3545 to swiggy@ybl on 28-12-2025")

print(result.amount)      # 2500.0
print(result.merchant)    # "Swiggy"
print(result.category)    # "food"
print(result.confidence)  # Confidence.HIGH
```

**Try it live:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ranjitbehera0034/Finance-Entity-Extractor/blob/main/examples/demo.ipynb)

---

## 📋 Output Schema Contract

Every extraction returns a guaranteed JSON structure:

```json
{
  "amount": 2500.0,           // float - Always numeric, never "Rs. 2,500"
  "currency": "INR",          // string - ISO 4217 code
  "type": "debit",            // string - "debit" | "credit"
  "account": "3545",          // string - Last 4 digits only
  "date": "28-12-2025",       // string - DD-MM-YYYY format
  "reference": "534567891234",// string - UPI/NEFT reference
  "merchant": "Swiggy",       // string - Normalized name (not "VPA-SWIGGY-BLR")
  "category": "food",         // string - Enum: food|shopping|transport|bills|...
  "vpa": "swiggy@ybl",        // string - Raw VPA
  "confidence": 0.95,         // float - 0.0 to 1.0
  "confidence_level": "HIGH"  // string - "LOW" | "MEDIUM" | "HIGH"
}
```

### Type Definitions (TypeScript-style)

```typescript
interface ExtractionResult {
  amount: number | null;
  currency: "INR";
  type: "debit" | "credit" | null;
  account: string | null;
  date: string | null;        // DD-MM-YYYY
  reference: string | null;
  merchant: string | null;
  category: Category | null;
  vpa: string | null;
  confidence: number;         // 0.0 - 1.0
  confidence_level: "LOW" | "MEDIUM" | "HIGH";
}

type Category = 
  | "food" | "shopping" | "transport" | "bills"
  | "entertainment" | "travel" | "grocery" | "fuel"
  | "healthcare" | "education" | "investment" | "transfer" | "other";
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

## 📊 Benchmark

| Metric | Value |
|--------|-------|
| Field Accuracy | 94.5% |
| Latency (Regex mode) | <1ms |
| Latency (LLM mode) | ~50ms |
| Throughput | 50,000+ msg/sec |

---

## 🔧 Installation Options

```bash
# Core (Regex + Rules only, no ML)
pip install finee

# With Apple Silicon backend
pip install "finee[metal]"

# With NVIDIA GPU backend
pip install "finee[cuda]"

# With CPU backend (llama.cpp)
pip install "finee[cpu]"
```

---

## 💻 CLI Usage

```bash
# Extract from text
finee extract "Rs.500 debited from A/c 1234"

# Check available backends
finee backends

# Show version
finee --version
```

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
│ TIER 1: Regex Engine                                        │
│ Extract: amount, date, reference, account, vpa, type        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: Rule-Based Mapping                                  │
│ Map: vpa → merchant, merchant → category                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: LLM (Optional, for missing fields)                  │
│ Targeted prompts for: merchant, category only               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 4: Validation + Normalization                          │
│ JSON repair, date normalization, confidence scoring         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
ExtractionResult (Guaranteed Schema)
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

[GitHub](https://github.com/Ranjitbehera0034/Finance-Entity-Extractor) · [PyPI](https://pypi.org/project/finee/) · [Hugging Face](https://huggingface.co/Ranjit0034/finance-entity-extractor)

</div>
