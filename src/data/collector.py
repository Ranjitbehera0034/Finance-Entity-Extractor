"""
Data Collection Orchestrator for Phase 2.

This module provides a complete pipeline for collecting, processing,
and preparing real financial data for model training.

Workflow:
    1. Collect PDFs from various sources
    2. Extract transactions using PDFExtractor
    3. Auto-label using EntityExtractor
    4. Human review and verification
    5. Export to training format

Data Sources:
    - Bank statement PDFs
    - PhonePe/GPay exports
    - Credit card statements
    - Email exports

Example:
    >>> from src.data.collector import DataCollector
    >>> collector = DataCollector("data/raw/pdfs")
    >>> collector.process_all()
    >>> collector.export_training_data("data/training/real_train.jsonl")

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

from src.data.pdf_extractor import PDFExtractor, Transaction, ExtractionResult, Bank
from src.data.extractor import EntityExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """
    Represents a data source for training data collection.
    
    Attributes:
        name: Unique identifier for the source.
        source_type: Type of source (pdf, email, csv).
        path: Path to source file or directory.
        bank: Associated bank (if applicable).
        processed: Whether source has been processed.
        transaction_count: Number of transactions extracted.
        last_processed: Timestamp of last processing.
    """
    
    name: str
    source_type: str  # pdf, email, csv, api
    path: Path
    bank: Optional[str] = None
    processed: bool = False
    transaction_count: int = 0
    last_processed: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass
class LabeledTransaction:
    """
    A transaction with labels for training.
    
    Attributes:
        source: Source identifier.
        raw_text: Original transaction text.
        subject: Generated or actual subject.
        entities: Extracted/labeled entities.
        verified: Whether human verified.
        confidence: Extraction confidence.
    """
    
    source: str
    raw_text: str
    subject: str
    entities: Dict[str, Any]
    verified: bool = False
    confidence: float = 0.0
    
    def to_training_format(self) -> Dict[str, str]:
        """Convert to training JSONL format."""
        prompt = f"Extract financial entities from this email:\n\nSubject: {self.subject}\n\nBody: {self.raw_text}"
        completion = json.dumps(self.entities, indent=2)
        return {"prompt": prompt, "completion": completion}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataCollector:
    """
    Orchestrates data collection and processing for training.
    
    This class manages the entire data collection pipeline:
    1. Source registration
    2. PDF extraction
    3. Auto-labeling
    4. Verification workflow
    5. Training data export
    
    Attributes:
        base_dir: Base directory for data storage.
        sources: Registered data sources.
        transactions: All extracted transactions.
        labeled_data: Processed and labeled data.
    
    Example:
        >>> collector = DataCollector("data/raw")
        >>> 
        >>> # Add sources
        >>> collector.add_pdf_folder("pdfs/hdfc", bank="hdfc")
        >>> collector.add_pdf_folder("pdfs/icici", bank="icici")
        >>> 
        >>> # Process
        >>> collector.process_all()
        >>> 
        >>> # Export
        >>> collector.export_training_data("data/training/real.jsonl")
    """
    
    def __init__(self, base_dir: Union[str, Path] = "data/raw") -> None:
        """
        Initialize the data collector.
        
        Args:
            base_dir: Base directory for raw data.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.pdf_dir = self.base_dir / "pdfs"
        self.email_dir = self.base_dir / "emails"
        self.export_dir = self.base_dir / "exports"
        
        for d in [self.pdf_dir, self.email_dir, self.export_dir]:
            d.mkdir(exist_ok=True)
        
        # State
        self.sources: List[DataSource] = []
        self.transactions: List[Transaction] = []
        self.labeled_data: List[LabeledTransaction] = []
        
        # Processing components
        self.pdf_extractor = PDFExtractor()
        self.entity_extractor = EntityExtractor()
        
        # State file
        self.state_file = self.base_dir / "collector_state.json"
        self._load_state()
        
        logger.info(f"DataCollector initialized: {self.base_dir}")
    
    def _load_state(self) -> None:
        """Load saved state if exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                
                # Restore sources
                for s in state.get("sources", []):
                    self.sources.append(DataSource(
                        name=s["name"],
                        source_type=s["source_type"],
                        path=Path(s["path"]),
                        bank=s.get("bank"),
                        processed=s.get("processed", False),
                        transaction_count=s.get("transaction_count", 0),
                        last_processed=s.get("last_processed"),
                    ))
                
                logger.info(f"Loaded state: {len(self.sources)} sources")
                
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self) -> None:
        """Save current state."""
        state = {
            "sources": [s.to_dict() for s in self.sources],
            "total_transactions": len(self.transactions),
            "total_labeled": len(self.labeled_data),
            "last_updated": datetime.now().isoformat(),
        }
        
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def add_pdf_folder(
        self, 
        folder: Union[str, Path],
        bank: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add a folder of PDFs as a data source.
        
        Args:
            folder: Path to folder containing PDFs.
            bank: Bank name for optimization.
            name: Unique source name.
        """
        folder = Path(folder)
        if not folder.exists():
            # Create if within base_dir
            if str(folder).startswith(str(self.base_dir)):
                folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created folder: {folder}")
            else:
                raise FileNotFoundError(f"Folder not found: {folder}")
        
        source_name = name or f"pdf_{folder.name}"
        
        # Check if already registered
        if any(s.name == source_name for s in self.sources):
            logger.warning(f"Source already registered: {source_name}")
            return
        
        source = DataSource(
            name=source_name,
            source_type="pdf",
            path=folder,
            bank=bank,
        )
        
        self.sources.append(source)
        self._save_state()
        
        logger.info(f"Added PDF source: {source_name} ({folder})")
    
    def add_pdf_file(
        self, 
        pdf_path: Union[str, Path],
        bank: Optional[str] = None
    ) -> None:
        """
        Add a single PDF file as a data source.
        
        Args:
            pdf_path: Path to PDF file.
            bank: Bank name.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        source_name = f"pdf_{pdf_path.stem}"
        
        source = DataSource(
            name=source_name,
            source_type="pdf",
            path=pdf_path,
            bank=bank,
        )
        
        self.sources.append(source)
        self._save_state()
        
        logger.info(f"Added PDF file: {pdf_path.name}")
    
    def process_all(self) -> int:
        """
        Process all registered data sources.
        
        Returns:
            Total number of transactions extracted.
        """
        total = 0
        
        for source in self.sources:
            if source.processed:
                logger.info(f"Skipping already processed: {source.name}")
                continue
            
            try:
                count = self._process_source(source)
                total += count
                source.processed = True
                source.transaction_count = count
                source.last_processed = datetime.now().isoformat()
                
            except Exception as e:
                source.errors.append(str(e))
                logger.error(f"Failed to process {source.name}: {e}")
        
        self._save_state()
        
        # Auto-label all transactions
        self._auto_label()
        
        logger.info(f"✅ Processed {total} transactions from {len(self.sources)} sources")
        return total
    
    def _process_source(self, source: DataSource) -> int:
        """Process a single data source."""
        logger.info(f"Processing: {source.name}")
        
        if source.source_type == "pdf":
            return self._process_pdf_source(source)
        else:
            logger.warning(f"Unknown source type: {source.source_type}")
            return 0
    
    def _process_pdf_source(self, source: DataSource) -> int:
        """Process PDF source."""
        bank = Bank(source.bank) if source.bank else None
        extractor = PDFExtractor(bank=bank)
        
        if source.path.is_file():
            result = extractor.extract(source.path)
            self.transactions.extend(result.transactions)
            return len(result.transactions)
        
        elif source.path.is_dir():
            total = 0
            pdf_files = list(source.path.glob("*.pdf")) + list(source.path.glob("*.PDF"))
            
            for pdf_file in pdf_files:
                try:
                    result = extractor.extract(pdf_file)
                    self.transactions.extend(result.transactions)
                    total += len(result.transactions)
                    logger.info(f"  {pdf_file.name}: {len(result.transactions)} transactions")
                except Exception as e:
                    logger.warning(f"  {pdf_file.name}: Error - {e}")
            
            return total
        
        return 0
    
    def _auto_label(self) -> None:
        """Auto-label all transactions using EntityExtractor."""
        logger.info(f"Auto-labeling {len(self.transactions)} transactions...")
        
        for txn in self.transactions:
            # Generate email-like text
            raw_text = self._transaction_to_email_text(txn)
            subject = f"{txn.bank.value.upper()} Transaction Alert"
            
            # Extract entities
            result = self.entity_extractor.extract(raw_text)
            confidence = result.confidence_score()
            
            # Use extracted or original data
            entities = {
                "amount": txn.amount,
                "type": txn.type.value,
            }
            
            if txn.balance:
                entities["balance"] = txn.balance
            if txn.reference:
                entities["reference"] = txn.reference
            if txn.category:
                entities["category"] = txn.category
            
            labeled = LabeledTransaction(
                source=f"pdf_{txn.bank.value}",
                raw_text=raw_text,
                subject=subject,
                entities=entities,
                verified=confidence >= 0.8,  # Auto-verify high confidence
                confidence=confidence,
            )
            
            self.labeled_data.append(labeled)
        
        verified_count = sum(1 for l in self.labeled_data if l.verified)
        logger.info(
            f"Labeled {len(self.labeled_data)} transactions "
            f"({verified_count} auto-verified)"
        )
    
    def _transaction_to_email_text(self, txn: Transaction) -> str:
        """Convert transaction to email-like text."""
        bank_name = txn.bank.value.upper()
        
        if txn.type.value == "debit":
            text = f"Dear Customer, Rs.{txn.amount} has been debited from your account"
        else:
            text = f"Dear Customer, Rs.{txn.amount} has been credited to your account"
        
        text += f" on {txn.date}."
        
        if txn.description:
            text += f" {txn.description}"
        
        if txn.reference:
            text += f" Ref: {txn.reference}."
        
        if txn.balance:
            text += f" Available balance: Rs.{txn.balance}."
        
        return text
    
    def get_pending_review(self) -> List[LabeledTransaction]:
        """Get transactions that need human review."""
        return [l for l in self.labeled_data if not l.verified]
    
    def verify_transaction(
        self, 
        index: int,
        corrected_entities: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Verify or correct a labeled transaction.
        
        Args:
            index: Index in labeled_data list.
            corrected_entities: Optional corrected entities.
        """
        if 0 <= index < len(self.labeled_data):
            if corrected_entities:
                self.labeled_data[index].entities = corrected_entities
            self.labeled_data[index].verified = True
            self.labeled_data[index].confidence = 1.0
    
    def export_training_data(
        self,
        output_file: Union[str, Path],
        verified_only: bool = True,
        train_split: float = 0.9
    ) -> Tuple[Path, Path]:
        """
        Export labeled data to training format.
        
        Args:
            output_file: Base path for output files.
            verified_only: Only export verified data.
            train_split: Train/validation split ratio.
        
        Returns:
            Tuple of (train_file, valid_file) paths.
        """
        data = self.labeled_data
        if verified_only:
            data = [l for l in data if l.verified]
        
        if not data:
            logger.warning("No data to export")
            return None, None
        
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        valid_data = data[split_idx:]
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path.parent / f"{output_path.stem}_train.jsonl"
        valid_file = output_path.parent / f"{output_path.stem}_valid.jsonl"
        
        for dataset, filepath in [(train_data, train_file), (valid_data, valid_file)]:
            with open(filepath, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item.to_training_format()) + "\n")
        
        logger.info(f"✅ Exported: {len(train_data)} train, {len(valid_data)} valid")
        
        return train_file, valid_file
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "sources": len(self.sources),
            "sources_processed": sum(1 for s in self.sources if s.processed),
            "total_transactions": len(self.transactions),
            "labeled_transactions": len(self.labeled_data),
            "verified": sum(1 for l in self.labeled_data if l.verified),
            "pending_review": sum(1 for l in self.labeled_data if not l.verified),
            "by_bank": self._count_by_bank(),
        }
    
    def _count_by_bank(self) -> Dict[str, int]:
        """Count transactions by bank."""
        counts: Dict[str, int] = {}
        for txn in self.transactions:
            bank = txn.bank.value
            counts[bank] = counts.get(bank, 0) + 1
        return counts
    
    def print_summary(self) -> None:
        """Print collection summary."""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print("📊 Data Collection Summary")
        print("=" * 50)
        print(f"Sources:        {stats['sources']} ({stats['sources_processed']} processed)")
        print(f"Transactions:   {stats['total_transactions']}")
        print(f"Labeled:        {stats['labeled_transactions']}")
        print(f"Verified:       {stats['verified']}")
        print(f"Pending Review: {stats['pending_review']}")
        print("\nBy Bank:")
        for bank, count in stats['by_bank'].items():
            print(f"  {bank.upper():10} {count}")
        print("=" * 50)


def main():
    """Example usage of DataCollector."""
    # Initialize collector
    collector = DataCollector("data/raw")
    
    # Show directory structure
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  📂 Data Collection Setup                    ║
╠══════════════════════════════════════════════════════════════╣
║  Place your files in the following directories:              ║
║                                                              ║
║  data/raw/                                                   ║
║  ├── pdfs/                                                   ║
║  │   ├── hdfc/         <- HDFC statements                   ║
║  │   ├── icici/        <- ICICI statements                  ║
║  │   ├── sbi/          <- SBI statements                    ║
║  │   └── other/        <- Other bank statements             ║
║  ├── exports/                                                ║
║  │   ├── phonepe/      <- PhonePe exports                   ║
║  │   └── gpay/         <- Google Pay exports                ║
║  └── emails/           <- Email exports (.mbox)             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create bank-specific folders
    banks = ["hdfc", "icici", "sbi", "axis", "kotak"]
    for bank in banks:
        (collector.pdf_dir / bank).mkdir(exist_ok=True)
    
    # Check for existing PDFs
    pdf_count = len(list(collector.pdf_dir.glob("**/*.pdf")))
    
    if pdf_count > 0:
        print(f"\n✅ Found {pdf_count} PDF files")
        
        # Register sources
        for bank_dir in collector.pdf_dir.iterdir():
            if bank_dir.is_dir():
                pdfs = list(bank_dir.glob("*.pdf"))
                if pdfs:
                    collector.add_pdf_folder(bank_dir, bank=bank_dir.name)
        
        # Process
        print("\n📝 Processing...")
        collector.process_all()
        
        # Summary
        collector.print_summary()
        
        # Export if we have data
        if collector.labeled_data:
            collector.export_training_data("data/training/real")
    else:
        print("\n⚠️  No PDF files found yet.")
        print("   Copy your bank statements to the folders above.")
        print("\nQuick commands:")
        print("   cp ~/Downloads/*.pdf data/raw/pdfs/hdfc/")


if __name__ == "__main__":
    main()
