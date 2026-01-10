# Indian Banking Email Samples for ML Training

A comprehensive collection of sample banking emails covering all transaction types across major Indian banks. Designed for training finance entity extraction models.

## Banks Covered
- HDFC Bank
- ICICI Bank
- SBI (State Bank of India)
- Axis Bank
- Kotak Mahindra Bank
- PNB (Punjab National Bank)

## Transaction Types

### Credit Types
| Type | Keywords |
|------|----------|
| UPI_CREDIT | credited, UPI, received |
| NEFT_CREDIT | NEFT, credited, transfer |
| RTGS_CREDIT | RTGS, credited |
| IMPS_CREDIT | IMPS, credited |
| SALARY_CREDIT | salary, credited |
| INTEREST_CREDIT | interest, credited |
| REFUND_CREDIT | refund, credited |
| CASH_DEPOSIT | cash, deposit, CDM |
| DIVIDEND_CREDIT | dividend, credited |

### Debit Types
| Type | Keywords |
|------|----------|
| UPI_DEBIT | debited, UPI, payment |
| NEFT_DEBIT | NEFT, debited, transfer |
| ATM_WITHDRAWAL | ATM, withdrawn, cash |
| POS_DEBIT | POS, merchant, debit card |
| BILL_PAYMENT | bill, payment, recharge |
| EMI_DEBIT | EMI, loan, deducted |
| SIP_DEBIT | SIP, mutual fund |

## Entity Extraction Fields

```json
{
  "date": "2026-01-10",
  "amount": 5000.00,
  "type": "credit|debit",
  "account": "4521",
  "bank": "hdfc",
  "reference": "503421789456",
  "merchant": "swiggy",
  "category": "food"
}
```

## VPA Patterns
- HDFC: @hdfcbank, @okhdfc
- ICICI: @icici, @okicici
- SBI: @sbi, @oksbi
- Axis: @axisbank, @okaxis
- Kotak: @kotak
- PayTM: @paytm
- PhonePe: @ybl
