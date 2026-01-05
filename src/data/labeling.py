"""
Semi-Automated Labeling Pipeline
Extracts entities using regex and allows human verification.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Import our extractor
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extractor import EntityExtractor


@dataclass
class LabeledExample:
    """A labeled training example."""
    id: int
    source: str  # 'email', 'pdf', 'manual'
    raw_text: str
    subject: str
    entities: Dict
    verified: bool = False
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_training_format(self) -> Dict:
        return {
            "prompt": f"Extract financial entities from this email:\n\nSubject: {self.subject}\n\nBody: {self.raw_text}",
            "completion": json.dumps(self.entities, indent=2)
        }


class LabelingPipeline:
    """
    Semi-automated labeling pipeline.
    
    Workflow:
    1. Load raw text (emails/PDFs)
    2. Auto-extract entities using regex
    3. Flag low-confidence extractions for review
    4. Human verifies/corrects
    5. Export training data
    """
    
    def __init__(self, data_dir: str = "data/labeling"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = EntityExtractor()
        self.examples: List[LabeledExample] = []
        self.next_id = 1
        
        # Load existing if any
        self._load_existing()
    
    def _load_existing(self):
        """Load existing labeled data."""
        labeled_file = self.data_dir / "labeled.json"
        if labeled_file.exists():
            with open(labeled_file) as f:
                data = json.load(f)
                self.examples = [LabeledExample(**ex) for ex in data]
                if self.examples:
                    self.next_id = max(ex.id for ex in self.examples) + 1
            print(f"📂 Loaded {len(self.examples)} existing examples")
    
    def add_raw_text(
        self, 
        text: str, 
        source: str = "manual",
        subject: str = ""
    ) -> LabeledExample:
        """Add raw text and auto-extract entities."""
        # Extract entities
        result = self.extractor.extract(text)
        entities = result.to_dict()
        
        # Calculate confidence based on how many fields extracted
        required_fields = ['amount', 'type']
        optional_fields = ['account', 'date', 'reference', 'merchant']
        
        required_score = sum(1 for f in required_fields if f in entities) / len(required_fields)
        optional_score = sum(1 for f in optional_fields if f in entities) / len(optional_fields)
        
        confidence = required_score * 0.6 + optional_score * 0.4
        
        example = LabeledExample(
            id=self.next_id,
            source=source,
            raw_text=text,
            subject=subject or self._generate_subject(entities),
            entities=entities,
            verified=confidence >= 0.8,  # Auto-verify high confidence
            confidence=confidence
        )
        
        self.examples.append(example)
        self.next_id += 1
        
        return example
    
    def _generate_subject(self, entities: Dict) -> str:
        """Generate a subject line from entities."""
        txn_type = entities.get('type', 'transaction').capitalize()
        amount = entities.get('amount', '')
        return f"{txn_type} Alert - Rs.{amount}"
    
    def add_emails(self, emails: List[Dict]) -> int:
        """Add emails from parsed data."""
        added = 0
        for email in emails:
            body = email.get('body', '')
            subject = email.get('subject', '')
            
            # Only add if it looks like a transaction
            if self._is_transaction_email(body):
                self.add_raw_text(body, source="email", subject=subject)
                added += 1
        
        print(f"📧 Added {added} transaction emails")
        return added
    
    def add_from_pdf(self, transactions: List[Dict]) -> int:
        """Add transactions extracted from PDF."""
        added = 0
        for txn in transactions:
            text = txn.get('raw_text', '') or txn.get('description', '')
            if text:
                self.add_raw_text(text, source="pdf")
                added += 1
        
        print(f"📄 Added {added} PDF transactions")
        return added
    
    def _is_transaction_email(self, text: str) -> bool:
        """Check if text is a transaction email."""
        text_lower = text.lower()
        has_amount = bool(re.search(r'rs\.?\s*[\d,]+|₹\s*[\d,]+', text_lower))
        has_txn_word = any(w in text_lower for w in ['debited', 'credited', 'paid', 'received'])
        return has_amount and has_txn_word
    
    def get_for_review(self) -> List[LabeledExample]:
        """Get examples that need human review."""
        return [ex for ex in self.examples if not ex.verified]
    
    def verify_example(self, example_id: int, corrected_entities: Dict = None):
        """Verify or correct an example."""
        for ex in self.examples:
            if ex.id == example_id:
                if corrected_entities:
                    ex.entities = corrected_entities
                ex.verified = True
                ex.confidence = 1.0
                return True
        return False
    
    def get_stats(self) -> Dict:
        """Get labeling statistics."""
        total = len(self.examples)
        verified = sum(1 for ex in self.examples if ex.verified)
        by_source = {}
        for ex in self.examples:
            by_source[ex.source] = by_source.get(ex.source, 0) + 1
        
        return {
            "total": total,
            "verified": verified,
            "pending_review": total - verified,
            "by_source": by_source,
            "avg_confidence": sum(ex.confidence for ex in self.examples) / total if total else 0
        }
    
    def save(self):
        """Save labeled data."""
        labeled_file = self.data_dir / "labeled.json"
        with open(labeled_file, "w") as f:
            json.dump([ex.to_dict() for ex in self.examples], f, indent=2)
        print(f"💾 Saved {len(self.examples)} examples to {labeled_file}")
    
    def export_training_data(
        self, 
        output_file: str = None,
        verified_only: bool = True,
        train_split: float = 0.9
    ) -> Tuple[Path, Path]:
        """Export to training format (JSONL)."""
        
        examples = [ex for ex in self.examples if ex.verified] if verified_only else self.examples
        
        if not examples:
            print("❌ No verified examples to export")
            return None, None
        
        # Shuffle
        import random
        random.shuffle(examples)
        
        # Split
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        valid_examples = examples[split_idx:]
        
        # Export
        output_dir = Path(output_file).parent if output_file else self.data_dir
        
        train_file = output_dir / "train.jsonl"
        valid_file = output_dir / "valid.jsonl"
        
        for examples_list, filepath in [(train_examples, train_file), (valid_examples, valid_file)]:
            with open(filepath, "w") as f:
                for ex in examples_list:
                    f.write(json.dumps(ex.to_training_format()) + "\n")
        
        print(f"✅ Exported: {len(train_examples)} train, {len(valid_examples)} valid")
        return train_file, valid_file
    
    def interactive_review(self):
        """Interactive CLI for reviewing examples."""
        pending = self.get_for_review()
        
        if not pending:
            print("✅ All examples are verified!")
            return
        
        print(f"\n📋 {len(pending)} examples need review")
        print("Commands: [y]es=verify, [e]dit, [s]kip, [q]uit\n")
        
        for i, ex in enumerate(pending):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(pending)}] ID: {ex.id} | Confidence: {ex.confidence:.1%}")
            print(f"Source: {ex.source}")
            print(f"\n📧 Text:\n{ex.raw_text[:300]}...")
            print(f"\n🔍 Extracted entities:")
            for k, v in ex.entities.items():
                print(f"   {k}: {v}")
            
            while True:
                choice = input("\nAction [y/e/s/q]: ").strip().lower()
                
                if choice == 'y':
                    self.verify_example(ex.id)
                    print("✅ Verified")
                    break
                elif choice == 'e':
                    print("Enter corrected JSON (or press Enter to cancel):")
                    try:
                        correction = input()
                        if correction:
                            corrected = json.loads(correction)
                            self.verify_example(ex.id, corrected)
                            print("✅ Corrected and verified")
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON")
                    break
                elif choice == 's':
                    print("⏭️ Skipped")
                    break
                elif choice == 'q':
                    self.save()
                    print(f"\n💾 Saved. Reviewed {i} examples.")
                    return
        
        self.save()
        print(f"\n✅ Review complete! All {len(pending)} examples processed.")


def main():
    """Example usage."""
    pipeline = LabelingPipeline()
    
    # Example: Add some test data
    test_emails = [
        "HDFC Bank: Rs.2500.00 has been debited from A/c **3545 on 28-12-25 to VPA swiggy@ybl (UPI Ref No 534567891234)",
        "Dear Customer, INR 45000 credited to A/c 7890 on 27-12-25. Salary from ACME Corp. Ref: 123456789",
        "SBI: Rs.1500 debited from a/c XX1234 on 26-12-25. UPI txn to amazon@apl. Ref:987654321",
    ]
    
    for email in test_emails:
        ex = pipeline.add_raw_text(email, source="test")
        print(f"\nAdded: {ex.subject}")
        print(f"  Entities: {ex.entities}")
        print(f"  Confidence: {ex.confidence:.1%}")
        print(f"  Auto-verified: {ex.verified}")
    
    # Show stats
    print(f"\n📊 Stats: {pipeline.get_stats()}")
    
    # Save
    pipeline.save()


if __name__ == "__main__":
    main()
