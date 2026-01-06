"""
Bank Statement Row Extractor for Phase 2.

Extracts text rows from bank statement PDFs and prepares them
for labeling and training. Uses pdfplumber for accurate text extraction.

Workflow:
    1. Extract raw rows from PDF tables
    2. Clean and normalize rows
    3. Export for manual labeling
    4. Generate synthetic variations
    5. Convert to training format with [BANK_STATEMENT] prefix

Banks Supported:
    - HDFC, ICICI, SBI, Axis, Kotak (Phase 2 target)

Example:
    >>> from src.data.statement_extractor import StatementRowExtractor
    >>> extractor = StatementRowExtractor()
    >>> rows = extractor.extract_rows("statement.pdf")
    >>> extractor.export_for_labeling(rows, "data/labeling/rows.json")

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class StatementRow:
    """
    Represents a single row from a bank statement.
    
    Attributes:
        raw_text: Original text from the PDF row.
        date: Transaction date (if detected).
        description: Transaction description/narration.
        debit: Debit amount (if applicable).
        credit: Credit amount (if applicable).
        balance: Running balance (if available).
        bank: Source bank.
        page: Page number in PDF.
        row_index: Row index on page.
        labeled: Whether manually labeled.
        entities: Labeled entities for training.
    """
    
    raw_text: str
    date: Optional[str] = None
    description: Optional[str] = None
    debit: Optional[str] = None
    credit: Optional[str] = None
    balance: Optional[str] = None
    bank: str = "unknown"
    page: int = 0
    row_index: int = 0
    labeled: bool = False
    entities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_training_format(self) -> Dict[str, str]:
        """
        Convert to training format with [BANK_STATEMENT] prefix.
        
        Returns:
            Dict with 'prompt' and 'completion' keys.
        """
        prefix = "[BANK_STATEMENT]"
        prompt = f"{prefix} Extract financial entities from this bank statement row:\n\n{self.raw_text}"
        
        entities = self.entities if self.entities else self._auto_entities()
        completion = json.dumps(entities, indent=2)
        
        return {"prompt": prompt, "completion": completion}
    
    def _auto_entities(self) -> Dict[str, Any]:
        """Generate entities from parsed fields."""
        entities = {}
        
        if self.date:
            entities["date"] = self.date
        if self.description:
            entities["description"] = self.description
        if self.debit:
            entities["amount"] = self.debit
            entities["type"] = "debit"
        elif self.credit:
            entities["amount"] = self.credit
            entities["type"] = "credit"
        if self.balance:
            entities["balance"] = self.balance
            
        return entities


@dataclass
class ExtractionStats:
    """Statistics for statement extraction."""
    total_pages: int = 0
    total_rows: int = 0
    valid_rows: int = 0
    extraction_time_seconds: float = 0.0
    bank: str = "unknown"
    errors: List[str] = field(default_factory=list)


class StatementRowExtractor:
    """
    Extracts and processes rows from bank statement PDFs.
    
    This class is specifically designed for Phase 2 of the model upgrade,
    focusing on bank statement row parsing rather than email extraction.
    
    Features:
        - Table-based row extraction using pdfplumber
        - Multi-bank format support
        - Row normalization and cleaning
        - Export for manual labeling
        - Synthetic data generation
        - Training data conversion
    
    Example:
        >>> extractor = StatementRowExtractor()
        >>> rows = extractor.extract_rows("hdfc_statement.pdf", bank="hdfc")
        >>> extractor.export_for_labeling(rows, "output.json")
    """
    
    # Date patterns for different banks
    DATE_PATTERNS = [
        r'\d{2}[-/]\d{2}[-/]\d{4}',  # DD-MM-YYYY, DD/MM/YYYY
        r'\d{2}[-/]\d{2}[-/]\d{2}',  # DD-MM-YY, DD/MM/YY
        r'\d{2}\s+[A-Za-z]{3}\s+\d{4}',  # 01 Jan 2025
        r'\d{2}\s+[A-Za-z]{3}\s+\d{2}',  # 01 Jan 25
        r'\d{4}[-/]\d{2}[-/]\d{2}',  # YYYY-MM-DD
    ]
    
    # Amount patterns
    AMOUNT_PATTERN = r'[\d,]+\.?\d*'
    
    # Column keywords by bank
    COLUMN_KEYWORDS = {
        "date": ["date", "txn date", "trans date", "value date", "transaction date"],
        "description": ["description", "narration", "particulars", "remarks", "details"],
        "debit": ["debit", "withdrawal", "dr", "withdrawals"],
        "credit": ["credit", "deposit", "cr", "deposits"],
        "balance": ["balance", "closing", "avl bal", "available"]
    }
    
    def __init__(self, debug: bool = False):
        """
        Initialize the statement row extractor.
        
        Args:
            debug: Enable debug logging.
        """
        self.debug = debug
        self._pdfplumber = None
        
        if debug:
            logger.setLevel(logging.DEBUG)
    
    @property
    def pdfplumber(self):
        """Lazy load pdfplumber."""
        if self._pdfplumber is None:
            try:
                import pdfplumber
                self._pdfplumber = pdfplumber
            except ImportError:
                raise ImportError(
                    "pdfplumber is required. Install with: pip install pdfplumber"
                )
        return self._pdfplumber
    
    def extract_rows(
        self,
        pdf_path: Union[str, Path],
        bank: Optional[str] = None,
        skip_header_rows: int = 1
    ) -> Tuple[List[StatementRow], ExtractionStats]:
        """
        Extract rows from a bank statement PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            bank: Bank name for optimized extraction.
            skip_header_rows: Number of header rows to skip in tables.
        
        Returns:
            Tuple of (list of StatementRow, ExtractionStats).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        start_time = datetime.now()
        stats = ExtractionStats()
        rows: List[StatementRow] = []
        
        # Detect bank if not provided
        detected_bank = bank or "unknown"
        
        try:
            with self.pdfplumber.open(pdf_path) as pdf:
                stats.total_pages = len(pdf.pages)
                
                # Try to detect bank from first page
                if detected_bank == "unknown":
                    first_page_text = pdf.pages[0].extract_text() or ""
                    detected_bank = self._detect_bank(first_page_text)
                
                stats.bank = detected_bank
                logger.info(f"Processing {pdf_path.name} ({detected_bank.upper()})")
                
                for page_idx, page in enumerate(pdf.pages):
                    page_rows = self._extract_page_rows(
                        page, 
                        page_idx + 1, 
                        detected_bank,
                        skip_header_rows
                    )
                    rows.extend(page_rows)
                    stats.total_rows += len(page_rows)
        
        except Exception as e:
            stats.errors.append(str(e))
            logger.error(f"Extraction failed: {e}")
        
        # Count valid rows
        stats.valid_rows = sum(1 for r in rows if r.date or r.description)
        stats.extraction_time_seconds = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Extracted {stats.valid_rows}/{stats.total_rows} valid rows")
        
        return rows, stats
    
    def _detect_bank(self, text: str) -> str:
        """Detect bank from text content."""
        text_lower = text.lower()
        
        bank_patterns = {
            "hdfc": ["hdfc bank", "hdfc ltd"],
            "icici": ["icici bank", "icici ltd"],
            "sbi": ["state bank of india", "sbi"],
            "axis": ["axis bank"],
            "kotak": ["kotak mahindra", "kotak bank"],
        }
        
        for bank, patterns in bank_patterns.items():
            if any(p in text_lower for p in patterns):
                return bank
        
        return "unknown"
    
    def _extract_page_rows(
        self, 
        page, 
        page_num: int, 
        bank: str,
        skip_header: int
    ) -> List[StatementRow]:
        """Extract rows from a single page."""
        rows = []
        
        # Try table extraction first
        tables = page.extract_tables()
        
        if tables:
            for table in tables:
                if not table or len(table) < 2:
                    continue
                
                # Parse table headers
                headers = [str(h).lower() if h else "" for h in table[0]]
                col_map = self._map_columns(headers)
                
                # Process data rows
                for row_idx, row in enumerate(table[skip_header:], start=1):
                    try:
                        statement_row = self._parse_table_row(
                            row, col_map, bank, page_num, row_idx
                        )
                        if statement_row:
                            rows.append(statement_row)
                    except Exception as e:
                        if self.debug:
                            logger.debug(f"Row parse error: {e}")
        
        else:
            # Fallback to text extraction
            text = page.extract_text()
            if text:
                for line_idx, line in enumerate(text.split('\n')):
                    if self._is_transaction_line(line):
                        row = self._parse_text_line(line, bank, page_num, line_idx)
                        if row:
                            rows.append(row)
        
        return rows
    
    def _map_columns(self, headers: List[str]) -> Dict[str, int]:
        """Map column indices from headers."""
        col_map = {}
        
        for col_type, keywords in self.COLUMN_KEYWORDS.items():
            for idx, header in enumerate(headers):
                if any(kw in header for kw in keywords):
                    col_map[col_type] = idx
                    break
        
        return col_map
    
    def _parse_table_row(
        self,
        row: List,
        col_map: Dict[str, int],
        bank: str,
        page_num: int,
        row_idx: int
    ) -> Optional[StatementRow]:
        """Parse a table row into StatementRow."""
        # Build raw text
        raw_text = " | ".join(str(c) if c else "" for c in row)
        
        # Skip if too short or no content
        if len(raw_text.strip()) < 10:
            return None
        
        # Extract fields
        date = self._get_cell(row, col_map.get("date"))
        description = self._get_cell(row, col_map.get("description"))
        debit = self._clean_amount(self._get_cell(row, col_map.get("debit")))
        credit = self._clean_amount(self._get_cell(row, col_map.get("credit")))
        balance = self._clean_amount(self._get_cell(row, col_map.get("balance")))
        
        # Skip if no amount
        if not (debit or credit):
            return None
        
        return StatementRow(
            raw_text=raw_text,
            date=date,
            description=description,
            debit=debit,
            credit=credit,
            balance=balance,
            bank=bank,
            page=page_num,
            row_index=row_idx
        )
    
    def _parse_text_line(
        self,
        line: str,
        bank: str,
        page_num: int,
        line_idx: int
    ) -> Optional[StatementRow]:
        """Parse a text line into StatementRow."""
        # Skip short lines
        if len(line.strip()) < 15:
            return None
        
        # Try to extract date
        date = None
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, line)
            if match:
                date = match.group()
                break
        
        # Try to extract amounts
        amounts = re.findall(self.AMOUNT_PATTERN, line)
        amounts = [a for a in amounts if len(a) > 3]  # Filter noise
        
        if not amounts:
            return None
        
        return StatementRow(
            raw_text=line.strip(),
            date=date,
            bank=bank,
            page=page_num,
            row_index=line_idx
        )
    
    def _get_cell(self, row: List, idx: Optional[int]) -> Optional[str]:
        """Safely get cell value."""
        if idx is not None and 0 <= idx < len(row):
            val = row[idx]
            return str(val).strip() if val else None
        return None
    
    def _clean_amount(self, value: Optional[str]) -> Optional[str]:
        """Clean and normalize amount value."""
        if not value:
            return None
        
        # Remove non-numeric except comma and decimal
        cleaned = re.sub(r'[^\d,.]', '', value)
        
        # Check if valid amount
        if cleaned and re.match(r'[\d,]+\.?\d*', cleaned):
            return cleaned
        
        return None
    
    def _is_transaction_line(self, line: str) -> bool:
        """Check if line looks like a transaction."""
        # Must have a date pattern
        has_date = any(re.search(p, line) for p in self.DATE_PATTERNS)
        
        # Must have amount-like numbers
        has_amount = bool(re.search(r'\d{1,3}(?:,\d{3})*\.?\d*', line))
        
        return has_date and has_amount
    
    def export_for_labeling(
        self,
        rows: List[StatementRow],
        output_path: Union[str, Path],
        include_metadata: bool = True
    ) -> Path:
        """
        Export rows for manual labeling.
        
        Args:
            rows: List of StatementRow objects.
            output_path: Output JSON file path.
            include_metadata: Include extraction metadata.
        
        Returns:
            Path to the output file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_rows": len(rows),
                "labeled": sum(1 for r in rows if r.labeled),
                "format_version": "1.0"
            },
            "rows": [r.to_dict() for r in rows]
        }
        
        if not include_metadata:
            data = data["rows"]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(rows)} rows to {output_path}")
        return output_path
    
    def load_labeled_data(self, input_path: Union[str, Path]) -> List[StatementRow]:
        """
        Load labeled data from JSON file.
        
        Args:
            input_path: Path to JSON file with labeled rows.
        
        Returns:
            List of StatementRow objects.
        """
        with open(input_path) as f:
            data = json.load(f)
        
        # Handle both formats
        rows_data = data.get("rows", data) if isinstance(data, dict) else data
        
        rows = []
        for r in rows_data:
            row = StatementRow(
                raw_text=r.get("raw_text", ""),
                date=r.get("date"),
                description=r.get("description"),
                debit=r.get("debit"),
                credit=r.get("credit"),
                balance=r.get("balance"),
                bank=r.get("bank", "unknown"),
                page=r.get("page", 0),
                row_index=r.get("row_index", 0),
                labeled=r.get("labeled", False),
                entities=r.get("entities", {})
            )
            rows.append(row)
        
        return rows


class StatementSyntheticGenerator:
    """
    Generate synthetic variations of bank statement rows.
    
    Creates training data by varying:
    - Amounts
    - Dates
    - Account numbers
    - Transaction descriptions
    - Balances
    """
    
    # Common transaction descriptions
    DESCRIPTIONS = {
        "upi": [
            "UPI-{merchant}@ybl-{name}",
            "UPI/{merchant}/{ref}",
            "UPI-TRANSFER-{merchant}",
            "IMPS/P2P/{name}",
        ],
        "neft": [
            "NEFT-{name}-{ref}",
            "NEFT CR-{bank}-{name}",
            "NEFT/TRANSFER/{account}",
        ],
        "card": [
            "POS {merchant} {city}",
            "ATM WDL {city}",
            "CARD TXN-{merchant}",
            "ECOM/{merchant}/ONLINE",
        ],
        "recurring": [
            "SB/AUTODR/{merchant}",
            "ECS/{merchant}/EMI",
            "AUTOPAY-{merchant}-{ref}",
        ]
    }
    
    MERCHANTS = [
        "Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Ola",
        "BigBasket", "Zepto", "PhonePe", "Paytm", "Netflix", "Spotify",
        "JioMart", "Myntra", "Nykaa", "BookMyShow", "MakeMyTrip"
    ]
    
    NAMES = [
        "Rahul Kumar", "Priya Sharma", "Amit Singh", "Neha Patel",
        "Vikram Reddy", "Anjali Gupta", "Ravi Verma", "Pooja Joshi"
    ]
    
    BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]
    CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad"]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        random.seed(seed)
    
    def generate_variations(
        self,
        base_rows: List[StatementRow],
        variations_per_row: int = 5,
        total_limit: Optional[int] = None
    ) -> List[StatementRow]:
        """
        Generate synthetic variations of labeled rows.
        
        Args:
            base_rows: Labeled rows to create variations from.
            variations_per_row: Number of variations per base row.
            total_limit: Maximum total rows to generate.
        
        Returns:
            List of synthetic StatementRow objects.
        """
        synthetic_rows = []
        
        for base_row in base_rows:
            for _ in range(variations_per_row):
                if total_limit and len(synthetic_rows) >= total_limit:
                    break
                
                variation = self._create_variation(base_row)
                synthetic_rows.append(variation)
            
            if total_limit and len(synthetic_rows) >= total_limit:
                break
        
        logger.info(f"Generated {len(synthetic_rows)} synthetic variations")
        return synthetic_rows
    
    def _create_variation(self, base: StatementRow) -> StatementRow:
        """Create a single variation of a base row."""
        # Generate new values
        new_amount = self._random_amount()
        new_date = self._random_date()
        new_balance = self._random_balance(new_amount)
        
        # Decide debit or credit
        is_debit = random.choice([True, False])
        
        # Generate description
        desc_type = random.choice(list(self.DESCRIPTIONS.keys()))
        template = random.choice(self.DESCRIPTIONS[desc_type])
        description = template.format(
            merchant=random.choice(self.MERCHANTS),
            name=random.choice(self.NAMES),
            ref=self._random_ref(),
            bank=random.choice(self.BANKS),
            account=self._random_account(),
            city=random.choice(self.CITIES)
        )
        
        # Create raw text based on bank format
        raw_text = self._format_row(
            date=new_date,
            description=description,
            amount=new_amount,
            is_debit=is_debit,
            balance=new_balance,
            bank=base.bank
        )
        
        # Build entities
        entities = {
            "date": new_date,
            "description": description,
            "amount": new_amount,
            "type": "debit" if is_debit else "credit",
            "balance": new_balance
        }
        
        return StatementRow(
            raw_text=raw_text,
            date=new_date,
            description=description,
            debit=new_amount if is_debit else None,
            credit=None if is_debit else new_amount,
            balance=new_balance,
            bank=base.bank,
            labeled=True,
            entities=entities
        )
    
    def _random_amount(self) -> str:
        """Generate random transaction amount."""
        # Various amount ranges
        ranges = [
            (10, 500),       # Small
            (500, 2000),     # Medium
            (2000, 10000),   # Large
            (10000, 50000),  # Very large
        ]
        
        min_val, max_val = random.choice(ranges)
        amount = random.uniform(min_val, max_val)
        
        # Format with optional decimals
        if random.random() < 0.3:
            return f"{amount:,.2f}"
        else:
            return f"{int(amount):,}"
    
    def _random_date(self) -> str:
        """Generate random date in various formats."""
        # Random date within last 180 days
        days_ago = random.randint(0, 180)
        date = datetime.now() - timedelta(days=days_ago)
        
        formats = [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d-%m-%y",
            "%d %b %Y",
            "%d %b %y",
        ]
        
        return date.strftime(random.choice(formats))
    
    def _random_balance(self, amount: str) -> str:
        """Generate random balance."""
        base = random.uniform(50000, 500000)
        return f"{base:,.2f}"
    
    def _random_ref(self) -> str:
        """Generate random reference number."""
        length = random.choice([8, 10, 12])
        return ''.join(str(random.randint(0, 9)) for _ in range(length))
    
    def _random_account(self) -> str:
        """Generate random account number suffix."""
        return ''.join(str(random.randint(0, 9)) for _ in range(4))
    
    def _format_row(
        self,
        date: str,
        description: str,
        amount: str,
        is_debit: bool,
        balance: str,
        bank: str
    ) -> str:
        """Format row based on bank style."""
        if is_debit:
            return f"{date} | {description} | {amount} | | {balance}"
        else:
            return f"{date} | {description} | | {amount} | {balance}"


def export_training_data(
    rows: List[StatementRow],
    output_path: Union[str, Path],
    train_split: float = 0.9
) -> Tuple[Path, Path]:
    """
    Export rows to training JSONL format.
    
    Args:
        rows: List of labeled StatementRow objects.
        output_path: Base path for output files.
        train_split: Train/validation split ratio.
    
    Returns:
        Tuple of (train_file, valid_file) paths.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter to labeled only
    labeled = [r for r in rows if r.labeled or r.entities]
    
    if not labeled:
        logger.warning("No labeled rows to export")
        return None, None
    
    # Shuffle and split
    random.shuffle(labeled)
    split_idx = int(len(labeled) * train_split)
    
    train_data = labeled[:split_idx]
    valid_data = labeled[split_idx:]
    
    # Output files
    train_file = output_path.parent / f"{output_path.stem}_train.jsonl"
    valid_file = output_path.parent / f"{output_path.stem}_valid.jsonl"
    
    for data, filepath in [(train_data, train_file), (valid_data, valid_file)]:
        with open(filepath, 'w') as f:
            for row in data:
                f.write(json.dumps(row.to_training_format()) + '\n')
    
    logger.info(f"✅ Exported: {len(train_data)} train, {len(valid_data)} valid")
    
    return train_file, valid_file


def main():
    """CLI for statement row extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract rows from bank statement PDFs"
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--bank", help="Bank name (auto-detected if not provided)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    extractor = StatementRowExtractor(debug=args.debug)
    rows, stats = extractor.extract_rows(args.pdf, bank=args.bank)
    
    print(f"\n📊 Extraction Complete")
    print(f"   Bank: {stats.bank.upper()}")
    print(f"   Pages: {stats.total_pages}")
    print(f"   Rows: {stats.valid_rows}/{stats.total_rows}")
    print(f"   Time: {stats.extraction_time_seconds:.2f}s")
    
    if args.output:
        extractor.export_for_labeling(rows, args.output)
    else:
        # Print sample rows
        print(f"\n📋 Sample Rows (first 5):")
        for row in rows[:5]:
            print(f"   {row.raw_text[:80]}...")


if __name__ == "__main__":
    main()
