---
license: mit
language:
- en
base_model: microsoft/Phi-3-mini-4k-instruct
pipeline_tag: text-generation
tags:
- finance
- entity-extraction
- email
- mlx
- lora
- phi-3
library_name: mlx
---

# 🧠 LLM Mail Trainer

> Fine-tune a local LLM to extract financial entities from personal emails using Apple Silicon (MLX).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-48%20passed-success.svg)

## ⚡ Quick Start

```python
# Install dependencies
# pip install mlx-lm

from mlx_lm import load, generate

# Load the model
model, tokenizer = load("Ranjit0034/finance-entity-extractor")

# Sample email
email = """
Dear Customer, Rs.2500.00 has been debited from account 3545 
to VPA swiggy@ybl on 28-12-25. Reference: 534567891234.
"""

# Extract entities
prompt = f"Extract financial entities from this email:\n\n{email}"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

**Output:**
```json
{
  "amount": "2500.00",
  "type": "debit",
  "account": "3545",
  "date": "28-12-25",
  "reference": "534567891234"
}
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
llm-mail-trainer/
├── 01_data_parsing.ipynb       # Parse MBOX → JSON
├── 01_data_pipeline.ipynb      # Full pipeline with classification
├── 02_classification.ipynb     # Email classification (Phi-3)
├── 03_pattern_discovery.ipynb  # Find patterns in finance emails
├── 04_training.ipynb           # LoRA fine-tuning
├── 05_add_credit_data.ipynb    # Augment with credit transactions
│
├── src/                        # 🆕 Python modules
│   ├── __init__.py
│   ├── data/
│   │   ├── parser.py           # Email parsing from MBOX
│   │   ├── extractor.py        # 🆕 Entity extraction with merchant/category
│   │   └── classifier.py       # 🆕 Rule-based & LLM classification
│   ├── training/
│   │   ├── prepare.py          # 🆕 Training data preparation
│   │   └── finetune.py         # 🆕 LoRA training wrapper
│   └── inference/
│       └── predict.py          # 🆕 CLI for entity extraction
│
├── config/
│   └── config.yaml             # Project configuration
├── data/
│   ├── raw/                    # Original MBOX file
│   ├── parsed/                 # Cleaned JSON emails
│   ├── filtered/               # Finance-specific emails
│   └── training/               # Train/valid JSONL files
├── models/
│   ├── base/phi3-mini/         # Base Phi-3 model
│   ├── adapters/               # LoRA adapters
│   └── merged/                 # Fused final model
├── tests/                      # 🆕 Unit tests (36 tests)
│   ├── test_entity_extraction.py
│   └── test_parser.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- Gmail MBOX export (Google Takeout)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-mail-trainer.git
cd llm-mail-trainer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download base model (Phi-3 Mini)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir models/base/phi3-mini
```

### Usage

1. **Export your Gmail data** via [Google Takeout](https://takeout.google.com/)
2. **Place MBOX file** in `data/raw/`
3. **Run notebooks in order**:
   ```bash
   jupyter notebook
   # Open 01_data_parsing.ipynb → 05_add_credit_data.ipynb
   ```

### 🌐 REST API (New!)

Start the API server:
```bash
cd llm-mail-trainer
source venv/bin/activate
python -m src.api.server
# Or: uvicorn src.api.server:app --reload
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/extract` | POST | Extract entities from email |
| `/classify` | POST | Classify email category |
| `/analyze` | POST | Full analysis (classify + extract) |
| `/batch` | POST | Process multiple emails |
| `/docs` | GET | Interactive Swagger docs |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"body": "Rs.500 debited from account 1234 on 01-01-25"}'
```

**Example Response:**
```json
{
  "success": true,
  "entities": {
    "amount": "500",
    "type": "debit",
    "account": "1234",
    "date": "01-01-25"
  },
  "extraction_time_ms": 0.5
}
```

## 📊 Sample Results

### Entity Extraction

**Input Email:**
```
Dear Customer, Rs.2500.00 has been debited from account 3545 
to VPA swiggy@ybl for Swiggy order on 28-12-25. 
Your UPI transaction reference number is 534567891234.
```

**Model Output (Enhanced):**
```json
{
  "amount": "2500.00",
  "type": "debit",
  "account": "3545",
  "date": "28-12-25",
  "reference": "534567891234",
  "merchant": "swiggy",
  "payment_method": "upi",
  "category": "food"
}
```

### Python Usage

```python
from src.data.extractor import EntityExtractor

extractor = EntityExtractor()
result = extractor.extract(email_body)

print(result.to_json())      # JSON output
print(result.merchant)       # "swiggy"
print(result.category)       # "food"
print(result.is_valid())     # True
```

### Project Stats

| Metric | Value |
|--------|-------|
| Total Emails Parsed | 40,820 |
| Finance Emails Identified | 8,116 (19.9%) |
| Synthetic Training Examples | 1,600 |
| Banks Supported | 8 (HDFC, ICICI, SBI, Axis, Kotak, GPay, Paytm, PhonePe) |
| Model Accuracy | **99.0%** ✨ |
| Unit Tests | 48 ✅ |

### 🏦 Phase 1 Evaluation Results (v3)

| Bank | Accuracy | Status |
|------|----------|--------|
| HDFC | 100% | ✅ |
| ICICI | 100% | ✅ |
| SBI | 100% | ✅ |
| Axis | 100% | ✅ |
| Kotak | 100% | ✅ |
| Paytm | 100% | ✅ |
| PhonePe | 100% | ✅ |
| GPay | 91.7% | ✅ |

## 🛠️ Technology Stack

- **[MLX](https://github.com/ml-explore/mlx)** - Apple Silicon ML framework
- **[mlx-lm](https://github.com/ml-explore/mlx-examples)** - LLM training/inference
- **[Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** - Base model (3.8B params)
- **[LoRA](https://arxiv.org/abs/2106.09685)** - Parameter-efficient fine-tuning
- **BeautifulSoup** - HTML parsing
- **Pandas** - Data manipulation

## 📈 Training Details

- **Base Model:** Phi-3 Mini 4K Instruct
- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA Layers:** 8
- **Iterations:** 500-600
- **Batch Size:** 1
- **Hardware:** Apple M-series chip

### Fine-tuning Command

```bash
mlx_lm.lora \
    --model models/base/phi3-mini \
    --data data/training \
    --train \
    --batch-size 1 \
    --lora-layers 8 \
    --iters 500 \
    --adapter-path models/adapters/finance-lora
```

## 🎯 Extracted Entities

| Entity | Description | Example |
|--------|-------------|---------|
| `amount` | Transaction amount | "2500.00" |
| `type` | Debit or Credit | "debit" |
| `account` | Account identifier | "3545" |
| `date` | Transaction date | "28-12-25" |
| `reference` | UPI/NEFT reference | "534567891234" |
| `merchant` | 🆕 Merchant name | "swiggy" |
| `payment_method` | 🆕 UPI/NEFT/Card/etc | "upi" |
| `category` | 🆕 Transaction category | "food" |

## 📝 Roadmap

### ✅ Phase 1: UPI Email Coverage (Complete)
- [x] Email parsing from MBOX
- [x] LLM-based classification
- [x] Pattern discovery
- [x] LoRA fine-tuning (600 iterations)
- [x] Credit transaction support (640 samples)
- [x] 8 banks supported with 200+ samples each
- [x] **99% accuracy** on held-out test set ✨
- [x] Unit tests (48 tests) ✨
- [x] REST API for inference ✨

### ✅ Phase 2: Bank Statement Parsing (Complete)
- [x] Bank statement PDF extraction
- [x] Text row parsing with pdfplumber
- [x] Synthetic 500 statement rows generated
- [x] [BANK_STATEMENT] prefix training
- [x] Retrained v4 model (800 iterations)
- [x] Dual capability: emails + statement rows

### 📋 Future Phases
- [ ] Spending analytics dashboard
- [ ] Real-time Gmail integration
- [ ] Docker deployment

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
