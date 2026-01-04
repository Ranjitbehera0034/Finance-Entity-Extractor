"""
Training Data Preparation Module
Converts parsed emails into training format for fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.extractor import EntityExtractor


@dataclass
class TrainingExample:
    """A single training example."""
    prompt: str
    completion: str
    
    def to_dict(self) -> Dict:
        return {"prompt": self.prompt, "completion": self.completion}
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())


class TrainingDataPreparer:
    """
    Prepare training data from parsed emails.
    
    Converts emails into prompt-completion pairs for fine-tuning.
    """
    
    # Prompt template for entity extraction task
    EXTRACTION_PROMPT_TEMPLATE = """Extract financial entities from this email:

Subject: {subject}

Body: {body}"""
    
    def __init__(
        self,
        emails_path: Path,
        output_dir: Path,
        extractor: Optional[EntityExtractor] = None
    ):
        """
        Initialize preparer.
        
        Args:
            emails_path: Path to parsed emails JSON
            output_dir: Directory to save training files
            extractor: EntityExtractor instance (created if not provided)
        """
        self.emails_path = Path(emails_path)
        self.output_dir = Path(output_dir)
        self.extractor = extractor or EntityExtractor()
        
        # Load emails
        with open(self.emails_path, 'r', encoding='utf-8') as f:
            self.all_emails = json.load(f)
        
        print(f"✅ Loaded {len(self.all_emails):,} emails")
    
    def filter_transaction_emails(self) -> List[Dict]:
        """Filter emails that contain transaction data."""
        transaction_emails = []
        
        for email in self.all_emails:
            body = email.get('body', '').lower()
            subject = email.get('subject', '').lower()
            combined = f"{subject} {body}"
            
            # Must have transaction indicators
            has_transaction = any(
                kw in combined 
                for kw in ['debited', 'credited', 'payment of', 'transferred']
            )
            
            # Must have amount pattern
            has_amount = 'rs' in combined or '₹' in combined
            
            if has_transaction and has_amount:
                transaction_emails.append(email)
        
        print(f"📧 Transaction emails found: {len(transaction_emails):,}")
        return transaction_emails
    
    def create_training_examples(
        self,
        emails: Optional[List[Dict]] = None,
        min_entities: int = 2,
        max_body_length: int = 1500
    ) -> List[TrainingExample]:
        """
        Convert emails to training examples.
        
        Args:
            emails: List of email dicts (uses transaction emails if None)
            min_entities: Minimum entities required for valid example
            max_body_length: Maximum body length in prompt
            
        Returns:
            List of TrainingExample objects
        """
        if emails is None:
            emails = self.filter_transaction_emails()
        
        examples = []
        skipped = 0
        
        for email in emails:
            # Extract entities
            entities = self.extractor.extract_to_dict(email.get('body', ''))
            
            # Skip if not enough entities
            if len(entities) < min_entities:
                skipped += 1
                continue
            
            # Create prompt
            prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
                subject=email.get('subject', '')[:200],
                body=email.get('body', '')[:max_body_length]
            )
            
            # Create completion (JSON output)
            completion = json.dumps(entities, indent=2)
            
            examples.append(TrainingExample(prompt=prompt, completion=completion))
        
        print(f"✅ Created {len(examples):,} training examples")
        print(f"⏭️ Skipped {skipped:,} (insufficient entities)")
        
        return examples
    
    def split_data(
        self,
        examples: List[TrainingExample],
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """
        Split examples into train and validation sets.
        
        Args:
            examples: List of training examples
            train_ratio: Ratio of examples for training (default 0.9)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_examples, valid_examples)
        """
        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        valid = shuffled[split_idx:]
        
        print(f"📊 Train: {len(train):,}, Validation: {len(valid):,}")
        
        return train, valid
    
    def save_jsonl(
        self,
        examples: List[TrainingExample],
        filename: str
    ) -> Path:
        """Save examples to JSONL file."""
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(example.to_jsonl() + '\n')
        
        print(f"💾 Saved {len(examples):,} examples to {output_path}")
        return output_path
    
    def prepare(
        self,
        train_ratio: float = 0.9,
        min_entities: int = 2,
        seed: int = 42
    ) -> Tuple[Path, Path]:
        """
        Full pipeline: filter → create examples → split → save.
        
        Returns:
            Tuple of (train_path, valid_path)
        """
        print("\n🚀 Starting training data preparation...")
        
        # Create examples
        examples = self.create_training_examples(min_entities=min_entities)
        
        if not examples:
            raise ValueError("No training examples created!")
        
        # Split data
        train, valid = self.split_data(examples, train_ratio, seed)
        
        # Save files
        train_path = self.save_jsonl(train, "train.jsonl")
        valid_path = self.save_jsonl(valid, "valid.jsonl")
        
        # Summary
        print("\n📋 Summary:")
        print(f"   Total examples: {len(examples):,}")
        print(f"   Train: {len(train):,}")
        print(f"   Valid: {len(valid):,}")
        print(f"   Output: {self.output_dir}")
        
        return train_path, valid_path
    
    def analyze_balance(self, examples: List[TrainingExample]) -> Dict[str, int]:
        """Analyze balance of transaction types in examples."""
        debit_count = sum(
            1 for e in examples 
            if '"type": "debit"' in e.completion
        )
        credit_count = sum(
            1 for e in examples 
            if '"type": "credit"' in e.completion
        )
        
        return {
            'debit': debit_count,
            'credit': credit_count,
            'other': len(examples) - debit_count - credit_count
        }


if __name__ == "__main__":
    from pathlib import Path
    
    PROJECT = Path.home() / "llm-mail-trainer"
    
    preparer = TrainingDataPreparer(
        emails_path=PROJECT / "data/parsed/emails.json",
        output_dir=PROJECT / "data/training"
    )
    
    train_path, valid_path = preparer.prepare()
    
    # Check balance
    examples = preparer.create_training_examples()
    balance = preparer.analyze_balance(examples)
    print(f"\n📊 Data Balance: {balance}")
