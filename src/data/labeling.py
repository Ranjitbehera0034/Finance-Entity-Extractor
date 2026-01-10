"""
Semi-Automated Labeling Pipeline.

Production-grade labeling system with regex auto-extraction,
confidence scoring, and interactive review capabilities.

Features:
    - Auto-extract entities using patterns
    - Confidence-based auto-verification
    - Interactive CLI for review
    - Batch processing
    - Export to training format

Example:
    >>> from src.data.labeling import LabelingPipeline
    >>> pipeline = LabelingPipeline()
    >>> 
    >>> # Add raw text
    >>> pipeline.add_text("Rs.500 debited from account 1234")
    >>> 
    >>> # Review pending
    >>> pipeline.interactive_review()
    >>> 
    >>> # Export
    >>> pipeline.export_training_data("data/training/labeled.jsonl")

Author: Ranjit Behera
License: MIT
"""

from __future__ import annotations

import json
import logging
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

from src.data.extractor import EntityExtractor, FinancialEntity

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LabeledExample:
    """
    A labeled training example with verification status.
    
    Attributes:
        id: Unique example identifier.
        source: Data source (email, pdf, manual).
        raw_text: Original text content.
        subject: Email subject or generated title.
        entities: Extracted/labeled entities.
        verified: Whether human verified.
        confidence: Auto-extraction confidence.
        created_at: Creation timestamp.
        verified_at: Verification timestamp.
        notes: Optional review notes.
    """
    
    id: int
    source: str
    raw_text: str
    subject: str
    entities: Dict[str, Any]
    verified: bool = False
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verified_at: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_training_format(self) -> Dict[str, str]:
        """Convert to training JSONL format."""
        prompt = (
            f"Extract financial entities from this email:\n\n"
            f"Subject: {self.subject}\n\n"
            f"Body: {self.raw_text}"
        )
        completion = json.dumps(self.entities, indent=2)
        return {"prompt": prompt, "completion": completion}
    
    def display(self) -> str:
        """Return formatted display string."""
        lines = [
            f"ID: {self.id} | Confidence: {self.confidence:.0%} | Verified: {self.verified}",
            f"Source: {self.source}",
            f"",
            f"📧 Subject: {self.subject}",
            f"",
            f"📝 Text:",
            f"   {self.raw_text[:300]}{'...' if len(self.raw_text) > 300 else ''}",
            f"",
            f"🔍 Entities:",
        ]
        for k, v in self.entities.items():
            lines.append(f"   {k}: {v}")
        
        return "\n".join(lines)


class LabelingPipeline:
    """
    Semi-automated labeling pipeline with interactive review.
    
    This pipeline automates entity extraction using regex patterns
    and allows human verification for low-confidence extractions.
    
    Workflow:
        1. Add raw text/emails/PDFs
        2. Auto-extract entities with confidence scoring
        3. High confidence (>80%) auto-verified
        4. Low confidence flagged for review
        5. Interactive CLI for human verification
        6. Export verified data to training format
    
    Attributes:
        data_dir: Directory for storing labeled data.
        extractor: EntityExtractor instance.
        examples: List of labeled examples.
    
    Example:
        >>> pipeline = LabelingPipeline("data/labeling")
        >>> 
        >>> # Add samples
        >>> pipeline.add_text(
        ...     "Rs.500 debited from A/c 1234 on 01-01-26",
        ...     source="email"
        ... )
        >>> 
        >>> # Check status
        >>> print(pipeline.get_stats())
        >>> 
        >>> # Review pending
        >>> pipeline.interactive_review()
    """
    
    # Confidence thresholds
    AUTO_VERIFY_THRESHOLD: float = 0.8
    HIGH_PRIORITY_THRESHOLD: float = 0.5
    
    def __init__(
        self, 
        data_dir: Union[str, Path] = "data/labeling",
        auto_load: bool = True
    ) -> None:
        """
        Initialize the labeling pipeline.
        
        Args:
            data_dir: Directory for labeled data storage.
            auto_load: Load existing data on init.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = EntityExtractor()
        self.examples: List[LabeledExample] = []
        self._next_id = 1
        
        # File paths
        self.data_file = self.data_dir / "labeled_data.json"
        self.export_dir = self.data_dir / "exports"
        self.export_dir.mkdir(exist_ok=True)
        
        if auto_load:
            self._load()
        
        logger.info(f"LabelingPipeline initialized: {self.data_dir}")
    
    def _load(self) -> None:
        """Load existing labeled data."""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    data = json.load(f)
                
                for item in data.get("examples", []):
                    self.examples.append(LabeledExample(**item))
                
                if self.examples:
                    self._next_id = max(e.id for e in self.examples) + 1
                
                logger.info(f"Loaded {len(self.examples)} examples")
                
            except Exception as e:
                logger.warning(f"Failed to load data: {e}")
    
    def _save(self) -> None:
        """Save labeled data to file."""
        data = {
            "examples": [e.to_dict() for e in self.examples],
            "stats": self.get_stats(),
            "last_updated": datetime.now().isoformat(),
        }
        
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved {len(self.examples)} examples")
    
    def add_text(
        self,
        text: str,
        source: str = "manual",
        subject: Optional[str] = None,
        auto_save: bool = True
    ) -> LabeledExample:
        """
        Add raw text for labeling.
        
        Text is auto-processed to extract entities and assigned
        a confidence score.
        
        Args:
            text: Raw transaction text.
            source: Source identifier (email, pdf, manual).
            subject: Email subject (auto-generated if None).
            auto_save: Save after adding.
        
        Returns:
            LabeledExample: The created example.
        """
        # Extract entities
        result = self.extractor.extract(text)
        entities = result.to_dict()
        confidence = result.confidence_score()
        
        # Generate subject if not provided
        if not subject:
            txn_type = entities.get("type", "Transaction").capitalize()
            amount = entities.get("amount", "")
            subject = f"{txn_type} Alert - Rs.{amount}"
        
        example = LabeledExample(
            id=self._next_id,
            source=source,
            raw_text=text.strip(),
            subject=subject,
            entities=entities,
            verified=confidence >= self.AUTO_VERIFY_THRESHOLD,
            confidence=confidence,
        )
        
        self.examples.append(example)
        self._next_id += 1
        
        if auto_save:
            self._save()
        
        logger.debug(f"Added example {example.id} (confidence: {confidence:.0%})")
        
        return example
    
    def add_batch(
        self,
        items: List[Dict[str, str]],
        source: str = "batch"
    ) -> int:
        """
        Add multiple items for labeling.
        
        Args:
            items: List of dicts with 'text' and optional 'subject'.
            source: Source identifier.
        
        Returns:
            Number of items added.
        """
        added = 0
        for item in items:
            text = item.get("text", "").strip()
            if text:
                self.add_text(
                    text=text,
                    source=source,
                    subject=item.get("subject"),
                    auto_save=False
                )
                added += 1
        
        self._save()
        logger.info(f"Added {added} examples from batch")
        
        return added
    
    def get_pending_review(self) -> List[LabeledExample]:
        """Get examples that need human review."""
        return [e for e in self.examples if not e.verified]
    
    def get_high_priority_review(self) -> List[LabeledExample]:
        """Get low-confidence examples needing review."""
        return [
            e for e in self.examples 
            if not e.verified and e.confidence < self.HIGH_PRIORITY_THRESHOLD
        ]
    
    def verify(
        self,
        example_id: int,
        corrected_entities: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Verify or correct an example.
        
        Args:
            example_id: Example ID to verify.
            corrected_entities: Optional corrected entities.
            notes: Optional notes about the verification.
        
        Returns:
            True if example found and verified.
        """
        for example in self.examples:
            if example.id == example_id:
                if corrected_entities:
                    example.entities = corrected_entities
                example.verified = True
                example.confidence = 1.0
                example.verified_at = datetime.now().isoformat()
                if notes:
                    example.notes = notes
                
                self._save()
                logger.info(f"Verified example {example_id}")
                return True
        
        return False
    
    def reject(self, example_id: int) -> bool:
        """
        Reject and remove an example.
        
        Args:
            example_id: Example to reject.
        
        Returns:
            True if found and removed.
        """
        for i, example in enumerate(self.examples):
            if example.id == example_id:
                del self.examples[i]
                self._save()
                logger.info(f"Rejected example {example_id}")
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get labeling statistics."""
        total = len(self.examples)
        verified = sum(1 for e in self.examples if e.verified)
        
        by_source: Dict[str, int] = {}
        for e in self.examples:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        
        return {
            "total": total,
            "verified": verified,
            "pending": total - verified,
            "auto_verified": sum(
                1 for e in self.examples 
                if e.verified and e.verified_at is None
            ),
            "human_verified": sum(
                1 for e in self.examples 
                if e.verified and e.verified_at is not None
            ),
            "avg_confidence": (
                sum(e.confidence for e in self.examples) / total 
                if total else 0
            ),
            "by_source": by_source,
        }
    
    def export_training_data(
        self,
        output_name: str = "labeled",
        verified_only: bool = True,
        train_split: float = 0.9
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Export to training format (JSONL).
        
        Args:
            output_name: Base name for output files.
            verified_only: Only export verified examples.
            train_split: Train/validation split ratio.
        
        Returns:
            Tuple of (train_path, valid_path).
        """
        data = self.examples
        if verified_only:
            data = [e for e in data if e.verified]
        
        if not data:
            logger.warning("No data to export")
            return None, None
        
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        valid_data = data[split_idx:]
        
        train_file = self.export_dir / f"{output_name}_train.jsonl"
        valid_file = self.export_dir / f"{output_name}_valid.jsonl"
        
        for dataset, filepath in [(train_data, train_file), (valid_data, valid_file)]:
            with open(filepath, "w") as f:
                for example in dataset:
                    f.write(json.dumps(example.to_training_format()) + "\n")
        
        logger.info(f"Exported: {len(train_data)} train, {len(valid_data)} valid")
        
        return train_file, valid_file
    
    def interactive_review(self, limit: Optional[int] = None) -> int:
        """
        Interactive CLI for reviewing pending examples.
        
        Args:
            limit: Maximum examples to review.
        
        Returns:
            Number of examples reviewed.
        """
        pending = self.get_pending_review()
        
        if not pending:
            print("\n✅ All examples are verified! Nothing to review.")
            return 0
        
        if limit:
            pending = pending[:limit]
        
        print(f"\n📋 Interactive Review Mode")
        print(f"   {len(pending)} examples pending review")
        print("\nCommands:")
        print("   [y] Verify as correct")
        print("   [e] Edit entities")
        print("   [r] Reject/remove")
        print("   [s] Skip")
        print("   [q] Quit")
        print("-" * 50)
        
        reviewed = 0
        
        for i, example in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}]")
            print("=" * 50)
            print(example.display())
            print("=" * 50)
            
            while True:
                choice = input("\nAction [y/e/r/s/q]: ").strip().lower()
                
                if choice == 'y':
                    self.verify(example.id)
                    print("✅ Verified")
                    reviewed += 1
                    break
                
                elif choice == 'e':
                    print("\nEnter corrected entities as JSON:")
                    print("Example: {\"amount\": \"500\", \"type\": \"debit\"}")
                    try:
                        json_input = input("> ").strip()
                        if json_input:
                            corrected = json.loads(json_input)
                            self.verify(example.id, corrected_entities=corrected)
                            print("✅ Corrected and verified")
                            reviewed += 1
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON, skipping")
                    break
                
                elif choice == 'r':
                    self.reject(example.id)
                    print("🗑️ Rejected")
                    reviewed += 1
                    break
                
                elif choice == 's':
                    print("⏭️ Skipped")
                    break
                
                elif choice == 'q':
                    print(f"\n💾 Saved. Reviewed {reviewed} examples.")
                    return reviewed
                
                else:
                    print("Invalid choice, try again")
        
        print(f"\n✅ Review complete! {reviewed} examples processed.")
        return reviewed
    
    def print_summary(self) -> None:
        """Print labeling summary."""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print("📊 Labeling Pipeline Summary")
        print("=" * 50)
        print(f"Total Examples:   {stats['total']}")
        print(f"Verified:         {stats['verified']} ({stats['verified']/stats['total']*100:.0f}%)" if stats['total'] else "Verified: 0")
        print(f"  Auto-verified:  {stats['auto_verified']}")
        print(f"  Human-verified: {stats['human_verified']}")
        print(f"Pending Review:   {stats['pending']}")
        print(f"Avg Confidence:   {stats['avg_confidence']:.0%}")
        print("\nBy Source:")
        for source, count in stats['by_source'].items():
            print(f"  {source:15} {count}")
        print("=" * 50)


if __name__ == "__main__":
    # Demo usage
    pipeline = LabelingPipeline()
    
    # Add sample transactions
    samples = [
        "HDFC Bank: Rs.2500.00 debited from A/c **3545 on 05-01-26 to VPA swiggy@ybl. Ref: 123456",
        "Dear Customer, INR 45000 credited to A/c 7890 on 04-01-26. Salary from ACME Corp.",
        "SBI: Rs.1500 debited from a/c XX1234 on 03-01-26. UPI txn to amazon@apl. Ref: 987654",
        "You paid Rs.599 to Uber from HDFC Bank. Txn ID: 456789.",
    ]
    
    print("Adding sample transactions...")
    for sample in samples:
        example = pipeline.add_text(sample, source="demo")
        print(f"  Added #{example.id}: confidence={example.confidence:.0%}")
    
    # Show summary
    pipeline.print_summary()
    
    # Export
    train, valid = pipeline.export_training_data()
    if train:
        print(f"\nExported to: {train}")
