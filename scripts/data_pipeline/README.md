# Data Pipeline

3-step pipeline to prepare training data for FinEE model fine-tuning.

## Overview

```
Raw Data (XML/JSON/CSV/MBOX)
         │
         ▼
┌────────────────────────┐
│ Step 1: Unify          │  → step1_unified.csv
│ Standardize formats    │     (all records, single schema)
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│ Step 2: Filter         │  → step2_training_ready.csv
│ Remove garbage         │     (clean transactions only)
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│ Step 3: Baseline       │  → step3_baseline_results.csv
│ Test current accuracy  │     (extracted fields + metrics)
└────────────────────────┘
```

## Quick Start

### 1. Copy your data to the workspace

```bash
# Example: Google Takeout export
cp -r ~/Downloads/Takeout /Users/ranjit/llm-mail-trainer/data/raw/

# Or SMS backup
cp sms_backup.xml /Users/ranjit/llm-mail-trainer/data/raw/
```

### 2. Run the pipeline

```bash
cd /Users/ranjit/llm-mail-trainer

# Step 1: Unify all data formats
python scripts/data_pipeline/step1_unify.py --input data/raw/ --output data/pipeline/step1_unified.csv

# Step 2: Filter garbage (OTPs, spam, etc.)
python scripts/data_pipeline/step2_filter.py --input data/pipeline/step1_unified.csv

# Step 3: Test baseline accuracy
python scripts/data_pipeline/step3_baseline.py
```

## Supported Input Formats

| Format | Source | Example |
|--------|--------|---------|
| `.mbox` | Gmail export | Mail.mbox |
| `.json` | Google Takeout | transactions.json |
| `.csv` | Bank exports | statements.csv |
| `.xml` | SMS Backup apps | sms_backup.xml |

## Output Schema

All data is standardized to:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | When message was received |
| `sender` | string | Bank/sender name |
| `body` | string | Message content |
| `source` | string | Original file source |

## Step 2 Filters

### Messages REMOVED (Garbage):
- OTPs / Verification codes
- Login alerts
- Marketing spam (% off, offers)
- Bill reminders (not transactions)
- Account statements
- Delivery notifications

### Messages KEPT (Transactions):
- Debit/Credit notifications
- UPI payments
- NEFT/IMPS transfers
- Amount + account references

## Step 3 Metrics

The baseline test measures:

| Metric | Description |
|--------|-------------|
| Extraction Success Rate | % of messages where amount + type extracted |
| Field Coverage | % for each field (amount, merchant, etc.) |
| Confidence Distribution | LOW / MEDIUM / HIGH breakdown |
| Top Merchants | Most common extracted merchants |
| Processing Speed | Messages per second |

## Directory Structure

```
data/
├── raw/                    # Put your raw data here
│   └── Takeout/
│       └── ...
├── pipeline/               # Pipeline outputs
│   ├── step1_unified.csv
│   ├── step2_training_ready.csv
│   ├── step2_garbage.csv       (optional)
│   ├── step2_uncertain.csv     (optional)
│   ├── step3_baseline_results.csv
│   └── step3_baseline_analysis.json
└── training/               # Final training data (after labeling)
```

## Next Steps

After Step 3:

1. Review low-confidence extractions
2. Add ground truth labels for training
3. Identify patterns that need new regex
4. Fine-tune the LLM on labeled data
