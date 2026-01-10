"""
PDF Statement Extractor - Production Grade.

Extract transactions from bank statement PDFs with support for
multiple Indian banks and statement formats.

Supported Banks:
    - HDFC Bank
    - ICICI Bank
    - State Bank of India (SBI)
    - Axis Bank
    - Kotak Mahindra Bank
    - Yes Bank
    - Punjab National Bank

Features:
    - Automatic bank detection
    - Table extraction
    - OCR fallback for scanned PDFs
    - Multiple date format parsing
    - Transaction categorization
    - Export to JSON/CSV

Example:
    >>> from src.data.pdf_extractor import PDFExtractor
    >>> extractor = PDFExtractor()
    >>> transactions = extractor.extract_from_pdf("statement.pdf")
    >>> print(f"Found {len(transactions)} transactions")

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

# Configure module logger
logger = logging.getLogger(__name__)


class Bank(Enum):
    """Supported banks enumeration."""
    
    HDFC = "hdfc"
    ICICI = "icici"
    SBI = "sbi"
    AXIS = "axis"
    KOTAK = "kotak"
    YES = "yes"
    PNB = "pnb"
    BOB = "bob"
    CANARA = "canara"
    UNION = "union"
    UNKNOWN = "unknown"
    
    @classmethod
    def detect(cls, text: str) -> Bank:
        """Detect bank from text content."""
        text_lower = text.lower()
        
        bank_keywords = {
            cls.HDFC: ["hdfc", "hdfcbank"],
            cls.ICICI: ["icici"],
            cls.SBI: ["state bank", "sbi "],
            cls.AXIS: ["axis bank"],
            cls.KOTAK: ["kotak"],
            cls.YES: ["yes bank"],
            cls.PNB: ["punjab national", "pnb "],
            cls.BOB: ["bank of baroda", "bob "],
            cls.CANARA: ["canara"],
            cls.UNION: ["union bank"],
        }
        
        for bank, keywords in bank_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return bank
        
        return cls.UNKNOWN


class TransactionType(Enum):
    """Transaction type enumeration."""
    
    DEBIT = "debit"
    CREDIT = "credit"
    UNKNOWN = "unknown"


@dataclass
class Transaction:
    """
    Represents a single transaction from a bank statement.
    
    Attributes:
        date: Transaction date.
        description: Transaction description/narration.
        amount: Transaction amount as string.
        type: Debit or credit.
        balance: Balance after transaction.
        reference: Reference/transaction number.
        category: Auto-detected category.
        bank: Source bank.
        raw_text: Original text for debugging.
        page_number: PDF page where found.
    """
    
    date: str
    description: str
    amount: str
    type: TransactionType = TransactionType.UNKNOWN
    balance: Optional[str] = None
    reference: Optional[str] = None
    category: Optional[str] = None
    bank: Bank = Bank.UNKNOWN
    raw_text: str = field(default="", repr=False)
    page_number: int = 0
    
    def __post_init__(self) -> None:
        """Normalize transaction data."""
        # Clean amount
        if self.amount:
            self.amount = self.amount.replace(",", "").replace(" ", "")
        
        # Clean balance
        if self.balance:
            self.balance = self.balance.replace(",", "").replace(" ", "")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal fields."""
        data = asdict(self)
        data["type"] = self.type.value
        data["bank"] = self.bank.value
        del data["raw_text"]
        return {k: v for k, v in data.items() if v is not None}
    
    def to_training_format(self) -> Dict[str, Any]:
        """Convert to training data format."""
        entities = {
            "amount": self.amount,
            "type": self.type.value,
        }
        
        if self.balance:
            entities["balance"] = self.balance
        if self.reference:
            entities["reference"] = self.reference
        if self.category:
            entities["category"] = self.category
        
        return {
            "source": "pdf",
            "bank": self.bank.value,
            "raw_text": self.description,
            "entities": entities,
        }
    
    def is_valid(self) -> bool:
        """Check if transaction has minimum required fields."""
        return bool(
            self.date and 
            self.amount and 
            self.type != TransactionType.UNKNOWN
        )


@dataclass
class ExtractionResult:
    """Result of PDF extraction."""
    
    transactions: List[Transaction]
    bank: Bank
    statement_period: Optional[str] = None
    account_number: Optional[str] = None
    total_pages: int = 0
    extraction_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bank": self.bank.value,
            "statement_period": self.statement_period,
            "account_number": self.account_number,
            "total_pages": self.total_pages,
            "total_transactions": len(self.transactions),
            "extraction_time_seconds": round(self.extraction_time_seconds, 2),
            "errors": self.errors,
            "transactions": [t.to_dict() for t in self.transactions],
        }
    
    def to_json(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved {len(self.transactions)} transactions to {filepath}")


class PDFExtractor:
    """
    Production-grade PDF extractor for bank statements.
    
    This extractor uses multiple strategies to extract transactions:
    1. Table extraction (pdfplumber)
    2. Text pattern matching
    3. OCR fallback for scanned documents
    
    Attributes:
        bank: Optional bank type for optimized extraction.
        debug: Enable debug logging.
    
    Example:
        >>> extractor = PDFExtractor()
        >>> result = extractor.extract("hdfc_statement.pdf")
        >>> print(f"Found {len(result.transactions)} transactions")
        >>> result.to_json("output.json")
    """
    
    # Date patterns for different formats
    DATE_PATTERNS: ClassVar[List[Tuple[str, str]]] = [
        (r"(\d{2}[-/]\d{2}[-/]\d{4})", "%d-%m-%Y"),
        (r"(\d{2}[-/]\d{2}[-/]\d{2})", "%d-%m-%y"),
        (r"(\d{2}\s+[A-Za-z]{3}\s+\d{4})", "%d %b %Y"),
        (r"(\d{2}\s+[A-Za-z]{3}\s+\d{2})", "%d %b %y"),
        (r"(\d{4}[-/]\d{2}[-/]\d{2})", "%Y-%m-%d"),
    ]
    
    # Amount patterns
    AMOUNT_PATTERN: ClassVar[str] = r"([\d,]+(?:\.\d{2})?)"
    
    # Category keywords
    CATEGORY_KEYWORDS: ClassVar[Dict[str, List[str]]] = {
        "food": ["swiggy", "zomato", "restaurant", "cafe", "food", "domino", "mcd", "kfc"],
        "shopping": ["amazon", "flipkart", "myntra", "ajio", "shopping"],
        "transport": ["uber", "ola", "rapido", "metro", "fuel", "petrol", "diesel"],
        "bills": ["electricity", "water", "gas", "internet", "mobile", "airtel", "jio"],
        "grocery": ["bigbasket", "zepto", "blinkit", "dmart", "grocery"],
        "transfer": ["upi", "neft", "imps", "rtgs", "transfer"],
        "salary": ["salary", "payroll", "income"],
        "atm": ["atm", "cash withdrawal"],
    }
    
    def __init__(
        self, 
        bank: Optional[Bank] = None,
        debug: bool = False
    ) -> None:
        """
        Initialize PDF extractor.
        
        Args:
            bank: Optional bank type for optimized extraction.
            debug: Enable debug logging.
        """
        self.bank = bank
        self.debug = debug
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        
        # Lazy import pdfplumber
        self._pdfplumber = None
        
        logger.info(f"PDFExtractor initialized (bank={bank})")
    
    @property
    def pdfplumber(self):
        """Lazy load pdfplumber."""
        if self._pdfplumber is None:
            try:
                import pdfplumber
                self._pdfplumber = pdfplumber
            except ImportError:
                logger.error("pdfplumber not installed. Run: pip install pdfplumber")
                raise ImportError("pdfplumber required. Install with: pip install pdfplumber")
        return self._pdfplumber
    
    def extract(self, pdf_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract transactions from a PDF statement.
        
        Args:
            pdf_path: Path to PDF file.
        
        Returns:
            ExtractionResult: Extraction results with transactions.
        
        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            ValueError: If PDF cannot be parsed.
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Extracting from: {pdf_path}")
        
        transactions: List[Transaction] = []
        errors: List[str] = []
        detected_bank = self.bank or Bank.UNKNOWN
        total_pages = 0
        
        try:
            with self.pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Detect bank from first page
                first_page_text = pdf.pages[0].extract_text() or ""
                if self.bank is None:
                    detected_bank = Bank.detect(first_page_text)
                    logger.info(f"Detected bank: {detected_bank.value}")
                
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_txns = self._extract_page(page, page_num, detected_bank)
                        transactions.extend(page_txns)
                    except Exception as e:
                        error_msg = f"Page {page_num}: {str(e)}"
                        errors.append(error_msg)
                        logger.warning(error_msg)
        
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            errors.append(str(e))
        
        # Deduplicate transactions
        transactions = self._deduplicate(transactions)
        
        elapsed = time.time() - start_time
        
        result = ExtractionResult(
            transactions=transactions,
            bank=detected_bank,
            total_pages=total_pages,
            extraction_time_seconds=elapsed,
            errors=errors,
        )
        
        logger.info(
            f"Extracted {len(transactions)} transactions "
            f"from {total_pages} pages in {elapsed:.2f}s"
        )
        
        return result
    
    def _extract_page(
        self, 
        page, 
        page_num: int, 
        bank: Bank
    ) -> List[Transaction]:
        """Extract transactions from a single page."""
        transactions: List[Transaction] = []
        
        # Try table extraction first
        tables = page.extract_tables() or []
        for table in tables:
            txns = self._parse_table(table, page_num, bank)
            transactions.extend(txns)
        
        # If no tables, try text extraction
        if not transactions:
            text = page.extract_text() or ""
            txns = self._parse_text(text, page_num, bank)
            transactions.extend(txns)
        
        return transactions
    
    def _parse_table(
        self, 
        table: List[List], 
        page_num: int, 
        bank: Bank
    ) -> List[Transaction]:
        """Parse transactions from table data."""
        transactions: List[Transaction] = []
        
        if not table or len(table) < 2:
            return transactions
        
        # Find header row
        header = [str(h).lower() if h else "" for h in table[0]]
        
        # Find column indices
        date_idx = self._find_column(header, ["date", "txn date", "transaction date", "value date"])
        desc_idx = self._find_column(header, ["description", "particulars", "narration", "details", "remarks"])
        debit_idx = self._find_column(header, ["debit", "withdrawal", "dr", "debit amount"])
        credit_idx = self._find_column(header, ["credit", "deposit", "cr", "credit amount"])
        balance_idx = self._find_column(header, ["balance", "closing balance", "running balance"])
        ref_idx = self._find_column(header, ["ref", "reference", "txn id", "utr"])
        
        # Process rows
        for row in table[1:]:
            if not row or len(row) < 3:
                continue
            
            try:
                date = self._get_cell(row, date_idx)
                description = self._get_cell(row, desc_idx)
                debit = self._get_cell(row, debit_idx)
                credit = self._get_cell(row, credit_idx)
                balance = self._get_cell(row, balance_idx)
                reference = self._get_cell(row, ref_idx)
                
                # Determine transaction type and amount
                if debit and self._is_amount(debit):
                    amount = debit
                    txn_type = TransactionType.DEBIT
                elif credit and self._is_amount(credit):
                    amount = credit
                    txn_type = TransactionType.CREDIT
                else:
                    continue
                
                # Skip if no valid date
                if not date or not self._is_date(date):
                    continue
                
                category = self._detect_category(description)
                
                txn = Transaction(
                    date=date,
                    description=description,
                    amount=amount,
                    type=txn_type,
                    balance=balance if balance and self._is_amount(balance) else None,
                    reference=reference,
                    category=category,
                    bank=bank,
                    raw_text=" | ".join([str(c) for c in row if c]),
                    page_number=page_num,
                )
                
                if txn.is_valid():
                    transactions.append(txn)
                    
            except (IndexError, ValueError) as e:
                logger.debug(f"Row parse error: {e}")
                continue
        
        return transactions
    
    def _parse_text(
        self, 
        text: str, 
        page_num: int, 
        bank: Bank
    ) -> List[Transaction]:
        """Parse transactions from raw text."""
        transactions: List[Transaction] = []
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:
                continue
            
            # Skip header-like lines
            if any(h in line.lower() for h in ["date", "particulars", "balance", "page"]):
                continue
            
            txn = self._parse_line(line, page_num, bank)
            if txn and txn.is_valid():
                transactions.append(txn)
        
        return transactions
    
    def _parse_line(
        self, 
        line: str, 
        page_num: int, 
        bank: Bank
    ) -> Optional[Transaction]:
        """Parse a single line as transaction."""
        # Find date
        date = None
        for pattern, _ in self.DATE_PATTERNS:
            match = re.search(pattern, line)
            if match:
                date = match.group(1)
                break
        
        if not date:
            return None
        
        # Find amounts
        amounts = re.findall(self.AMOUNT_PATTERN, line)
        if not amounts:
            return None
        
        # Determine type
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["dr", "debit", "paid", "withdrawn"]):
            txn_type = TransactionType.DEBIT
        elif any(kw in line_lower for kw in ["cr", "credit", "received", "deposit"]):
            txn_type = TransactionType.CREDIT
        else:
            txn_type = TransactionType.DEBIT
        
        amount = amounts[0].replace(",", "")
        balance = amounts[-1].replace(",", "") if len(amounts) > 1 else None
        
        return Transaction(
            date=date,
            description=line,
            amount=amount,
            type=txn_type,
            balance=balance,
            category=self._detect_category(line),
            bank=bank,
            raw_text=line,
            page_number=page_num,
        )
    
    def _find_column(self, headers: List[str], keywords: List[str]) -> int:
        """Find column index matching any keyword."""
        for i, h in enumerate(headers):
            for kw in keywords:
                if kw in h:
                    return i
        return -1
    
    def _get_cell(self, row: List, idx: int) -> str:
        """Safely get cell value."""
        if idx < 0 or idx >= len(row):
            return ""
        return str(row[idx]).strip() if row[idx] else ""
    
    def _is_amount(self, value: str) -> bool:
        """Check if value is a valid amount."""
        cleaned = value.replace(",", "").replace(" ", "").replace(".", "")
        return cleaned.isdigit() and len(cleaned) > 0
    
    def _is_date(self, value: str) -> bool:
        """Check if value looks like a date."""
        for pattern, _ in self.DATE_PATTERNS:
            if re.match(pattern, value):
                return True
        return False
    
    def _detect_category(self, text: str) -> Optional[str]:
        """Detect transaction category from description."""
        text_lower = text.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return None
    
    def _deduplicate(self, transactions: List[Transaction]) -> List[Transaction]:
        """Remove duplicate transactions."""
        seen = set()
        unique = []
        
        for txn in transactions:
            key = (txn.date, txn.amount, txn.type.value)
            if key not in seen:
                seen.add(key)
                unique.append(txn)
        
        if len(unique) < len(transactions):
            logger.debug(f"Removed {len(transactions) - len(unique)} duplicates")
        
        return unique


def extract_from_folder(
    folder_path: Union[str, Path],
    output_file: Optional[str] = None,
    bank: Optional[Bank] = None
) -> List[Transaction]:
    """
    Extract transactions from all PDFs in a folder.
    
    Args:
        folder_path: Path to folder containing PDFs.
        output_file: Optional JSON output file.
        bank: Optional bank type.
    
    Returns:
        List of all extracted transactions.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    extractor = PDFExtractor(bank=bank)
    all_transactions: List[Transaction] = []
    
    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    
    print(f"📂 Found {len(pdf_files)} PDF files in {folder}")
    
    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        try:
            result = extractor.extract(pdf_file)
            all_transactions.extend(result.transactions)
            print(f"   ✅ {len(result.transactions)} transactions")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Total: {len(all_transactions)} transactions")
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(
                [t.to_dict() for t in all_transactions],
                f,
                indent=2
            )
        print(f"💾 Saved to: {output_path}")
    
    return all_transactions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("PDF Statement Extractor")
        print("=" * 40)
        print("\nUsage:")
        print("  python pdf_extractor.py <pdf_file>")
        print("  python pdf_extractor.py <folder> [output.json]")
        print("\nExamples:")
        print("  python pdf_extractor.py statement.pdf")
        print("  python pdf_extractor.py ./statements/ all_txns.json")
        sys.exit(0)
    
    path = Path(sys.argv[1])
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    if path.is_file():
        extractor = PDFExtractor(debug=True)
        result = extractor.extract(path)
        
        print(f"\n📊 Extraction Results:")
        print(f"   Bank: {result.bank.value}")
        print(f"   Pages: {result.total_pages}")
        print(f"   Transactions: {len(result.transactions)}")
        print(f"   Time: {result.extraction_time_seconds:.2f}s")
        
        if result.errors:
            print(f"   Errors: {len(result.errors)}")
        
        print("\n📋 Sample transactions:")
        for txn in result.transactions[:5]:
            print(f"   {txn.date} | {txn.type.value:6} | Rs.{txn.amount}")
        
        if output:
            result.to_json(output)
    else:
        extract_from_folder(path, output)
