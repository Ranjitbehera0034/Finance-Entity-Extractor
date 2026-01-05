"""
PDF Text Extractor for Bank Statements
Extracts transaction text from PDF bank statements.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import os
    os.system("pip install pdfplumber -q")
    import pdfplumber


@dataclass
class Transaction:
    """Extracted transaction from PDF."""
    date: str
    description: str
    amount: str
    type: str  # debit/credit
    balance: Optional[str] = None
    reference: Optional[str] = None
    raw_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "type": self.type,
            "balance": self.balance,
            "reference": self.reference,
            "raw_text": self.raw_text
        }


class PDFExtractor:
    """Extract transactions from bank statement PDFs."""
    
    # Bank-specific patterns
    BANK_PATTERNS = {
        "hdfc": {
            "date": r"(\d{2}/\d{2}/\d{2,4})",
            "amount": r"([\d,]+\.\d{2})",
            "type_debit": r"(DR|Debit|DEBIT)",
            "type_credit": r"(CR|Credit|CREDIT)",
        },
        "icici": {
            "date": r"(\d{2}-\d{2}-\d{4})",
            "amount": r"([\d,]+\.\d{2})",
            "type_debit": r"(Dr\.?|Debit)",
            "type_credit": r"(Cr\.?|Credit)",
        },
        "sbi": {
            "date": r"(\d{2}\s+\w{3}\s+\d{4})",
            "amount": r"([\d,]+\.\d{2})",
            "type_debit": r"(Debit|DR)",
            "type_credit": r"(Credit|CR)",
        },
        "axis": {
            "date": r"(\d{2}-\d{2}-\d{4})",
            "amount": r"([\d,]+\.\d{2})",
            "type_debit": r"(DR|Debit)",
            "type_credit": r"(CR|Credit)",
        },
        "kotak": {
            "date": r"(\d{2}/\d{2}/\d{4})",
            "amount": r"([\d,]+\.\d{2})",
            "type_debit": r"(Dr|Debit)",
            "type_credit": r"(Cr|Credit)",
        },
    }
    
    # UPI keywords
    UPI_KEYWORDS = [
        "upi", "vpa", "@", "imps", "neft", "rtgs",
        "phonepe", "gpay", "paytm", "bhim"
    ]
    
    def __init__(self, bank: str = "auto"):
        """
        Initialize extractor.
        
        Args:
            bank: Bank name (hdfc, icici, sbi, axis, kotak) or 'auto'
        """
        self.bank = bank.lower()
    
    def extract_from_pdf(self, pdf_path: str) -> List[Transaction]:
        """Extract all transactions from a PDF."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        transactions = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                
                # Try table extraction first
                for table in tables:
                    txns = self._parse_table(table)
                    transactions.extend(txns)
                
                # Also try text extraction
                txns = self._parse_text(text)
                transactions.extend(txns)
        
        # Deduplicate
        seen = set()
        unique = []
        for txn in transactions:
            key = (txn.date, txn.amount, txn.type)
            if key not in seen:
                seen.add(key)
                unique.append(txn)
        
        print(f"📄 Extracted {len(unique)} transactions from {pdf_path.name}")
        return unique
    
    def _parse_table(self, table: List[List]) -> List[Transaction]:
        """Parse transactions from table data."""
        transactions = []
        
        if not table or len(table) < 2:
            return transactions
        
        # Try to find header row
        header = table[0] if table[0] else []
        header_lower = [str(h).lower() if h else "" for h in header]
        
        # Find column indices
        date_idx = self._find_column(header_lower, ["date", "txn date", "transaction date"])
        desc_idx = self._find_column(header_lower, ["description", "particulars", "narration", "details"])
        debit_idx = self._find_column(header_lower, ["debit", "withdrawal", "dr"])
        credit_idx = self._find_column(header_lower, ["credit", "deposit", "cr"])
        balance_idx = self._find_column(header_lower, ["balance", "closing"])
        
        for row in table[1:]:
            if not row or len(row) < 3:
                continue
            
            try:
                date = str(row[date_idx]) if date_idx >= 0 and row[date_idx] else ""
                desc = str(row[desc_idx]) if desc_idx >= 0 and row[desc_idx] else ""
                
                # Determine if debit or credit
                debit_amt = str(row[debit_idx]) if debit_idx >= 0 and row[debit_idx] else ""
                credit_amt = str(row[credit_idx]) if credit_idx >= 0 and row[credit_idx] else ""
                
                if debit_amt and debit_amt.replace(",", "").replace(".", "").isdigit():
                    amount = debit_amt.replace(",", "")
                    txn_type = "debit"
                elif credit_amt and credit_amt.replace(",", "").replace(".", "").isdigit():
                    amount = credit_amt.replace(",", "")
                    txn_type = "credit"
                else:
                    continue
                
                balance = str(row[balance_idx]) if balance_idx >= 0 and row[balance_idx] else None
                
                if date and amount:
                    transactions.append(Transaction(
                        date=date.strip(),
                        description=desc.strip(),
                        amount=amount,
                        type=txn_type,
                        balance=balance,
                        raw_text=" | ".join([str(c) for c in row if c])
                    ))
            except (IndexError, ValueError):
                continue
        
        return transactions
    
    def _parse_text(self, text: str) -> List[Transaction]:
        """Parse transactions from raw text."""
        transactions = []
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:
                continue
            
            # Skip header lines
            if any(h in line.lower() for h in ["date", "particular", "balance", "page"]):
                continue
            
            # Try to extract transaction
            txn = self._extract_from_line(line)
            if txn:
                transactions.append(txn)
        
        return transactions
    
    def _extract_from_line(self, line: str) -> Optional[Transaction]:
        """Extract transaction from a single line."""
        # Date pattern
        date_match = re.search(r"(\d{2}[-/]\d{2}[-/]\d{2,4})", line)
        if not date_match:
            return None
        
        # Amount pattern
        amounts = re.findall(r"([\d,]+\.\d{2})", line)
        if not amounts:
            return None
        
        # Determine type
        line_lower = line.lower()
        if "debited" in line_lower or "dr" in line_lower or "paid" in line_lower:
            txn_type = "debit"
        elif "credited" in line_lower or "cr" in line_lower or "received" in line_lower:
            txn_type = "credit"
        else:
            txn_type = "debit"  # Default to debit
        
        # Get the main amount (usually the larger or first significant one)
        amount = amounts[0].replace(",", "")
        
        return Transaction(
            date=date_match.group(1),
            description=line,
            amount=amount,
            type=txn_type,
            raw_text=line
        )
    
    def _find_column(self, headers: List[str], keywords: List[str]) -> int:
        """Find column index matching any keyword."""
        for i, h in enumerate(headers):
            for kw in keywords:
                if kw in h:
                    return i
        return -1
    
    def is_upi_transaction(self, txn: Transaction) -> bool:
        """Check if transaction is UPI-related."""
        text = (txn.description + " " + txn.raw_text).lower()
        return any(kw in text for kw in self.UPI_KEYWORDS)
    
    def filter_upi_only(self, transactions: List[Transaction]) -> List[Transaction]:
        """Filter to only UPI transactions."""
        upi_txns = [t for t in transactions if self.is_upi_transaction(t)]
        print(f"🔍 Found {len(upi_txns)} UPI transactions out of {len(transactions)}")
        return upi_txns


def extract_statements(pdf_folder: str, output_file: str = None):
    """Extract transactions from all PDFs in a folder."""
    import json
    
    folder = Path(pdf_folder)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return []
    
    extractor = PDFExtractor()
    all_transactions = []
    
    for pdf_file in folder.glob("*.pdf"):
        print(f"\n📄 Processing: {pdf_file.name}")
        try:
            txns = extractor.extract_from_pdf(pdf_file)
            all_transactions.extend(txns)
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
    
    print(f"\n✅ Total transactions extracted: {len(all_transactions)}")
    
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump([t.to_dict() for t in all_transactions], f, indent=2)
        print(f"💾 Saved to: {output_path}")
    
    return all_transactions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_file_or_folder> [output.json]")
        print("\nExample:")
        print("  python pdf_extractor.py ~/statements/hdfc.pdf")
        print("  python pdf_extractor.py ~/statements/ all_transactions.json")
    else:
        path = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None
        
        if Path(path).is_file():
            extractor = PDFExtractor()
            txns = extractor.extract_from_pdf(path)
            for t in txns[:10]:
                print(f"  {t.date} | {t.type:6} | Rs.{t.amount}")
        else:
            extract_statements(path, output)
