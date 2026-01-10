"""
Inference/Prediction Module
Load fine-tuned model and extract entities from emails.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Result of model prediction."""
    entities: Dict
    raw_response: str
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "entities": self.entities,
            "raw_response": self.raw_response,
            "success": self.success,
            "error": self.error
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.entities, indent=indent)


class Predictor:
    """
    Load and run inference with fine-tuned model.
    
    Supports both:
    - Base model + LoRA adapter
    - Merged model
    """
    
    EXTRACTION_PROMPT = """Extract financial entities from this email:

Subject: {subject}

Body: {body}"""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        adapter_path: Optional[Union[str, Path]] = None,
        max_tokens: int = 200
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model (base or merged)
            adapter_path: Optional path to LoRA adapter
            max_tokens: Maximum tokens to generate
        """
        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.max_tokens = max_tokens
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return
        
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "mlx_lm is required. Install with: pip install mlx-lm"
            )
        
        print(f"🔄 Loading model from {self.model_path}...")
        
        if self.adapter_path:
            print(f"   With adapter: {self.adapter_path}")
            self.model, self.tokenizer = load(
                str(self.model_path),
                adapter_path=str(self.adapter_path)
            )
        else:
            self.model, self.tokenizer = load(str(self.model_path))
        
        self._loaded = True
        print("✅ Model loaded successfully!")
    
    def predict(
        self,
        subject: str = "",
        body: str = "",
        email_text: Optional[str] = None
    ) -> PredictionResult:
        """
        Extract entities from an email.
        
        Args:
            subject: Email subject
            body: Email body
            email_text: Full email text (alternative to subject+body)
            
        Returns:
            PredictionResult with extracted entities
        """
        if not self._loaded:
            self.load()
        
        try:
            from mlx_lm import generate
        except ImportError:
            raise ImportError("mlx_lm is required")
        
        # Build prompt
        if email_text:
            prompt = f"Extract financial entities from this email:\n\n{email_text}"
        else:
            prompt = self.EXTRACTION_PROMPT.format(
                subject=subject[:200],
                body=body[:1500]
            )
        
        # Generate response
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False
            )
        except Exception as e:
            return PredictionResult(
                entities={},
                raw_response="",
                success=False,
                error=f"Generation failed: {str(e)}"
            )
        
        # Parse JSON from response
        entities = self._extract_json(response)
        
        return PredictionResult(
            entities=entities if entities else {},
            raw_response=response,
            success=entities is not None
        )
    
    def predict_batch(
        self,
        emails: list
    ) -> list:
        """
        Extract entities from multiple emails.
        
        Args:
            emails: List of dicts with 'subject' and 'body' keys
            
        Returns:
            List of PredictionResults
        """
        results = []
        for email in emails:
            result = self.predict(
                subject=email.get('subject', ''),
                body=email.get('body', '')
            )
            results.append(result)
        return results
    
    def _extract_json(self, response: str) -> Optional[Dict]:
        """Extract JSON object from model response."""
        # Find JSON pattern
        match = re.search(r'\{[^{}]*\}', response)
        
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        return None


def main():
    """CLI for running predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract financial entities from emails using fine-tuned LLM"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path.home() / "llm-mail-trainer/models/base/phi3-mini"),
        help="Path to model"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Email subject"
    )
    parser.add_argument(
        "--body",
        type=str,
        default=None,
        help="Email body text"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to file containing email text"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = Predictor(
        model_path=args.model,
        adapter_path=args.adapter
    )
    
    if args.interactive:
        run_interactive(predictor)
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read()
        predictor.load()
        result = predictor.predict(email_text=text)
        print(result.to_json())
    elif args.body:
        predictor.load()
        result = predictor.predict(subject=args.subject, body=args.body)
        print(result.to_json())
    else:
        parser.print_help()


def run_interactive(predictor: Predictor):
    """Interactive mode for testing."""
    predictor.load()
    
    print("\n" + "=" * 60)
    print("🧠 LLM Mail Trainer - Interactive Mode")
    print("=" * 60)
    print("Enter email text to extract entities.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    while True:
        print("\n📧 Enter email text (multi-line, end with empty line):")
        lines = []
        while True:
            try:
                line = input()
                if line.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    return
                if line == "" and lines:
                    break
                lines.append(line)
            except EOFError:
                print("\n👋 Goodbye!")
                return
        
        email_text = "\n".join(lines)
        
        if email_text.strip():
            print("\n🔄 Extracting entities...")
            result = predictor.predict(email_text=email_text)
            
            print("\n📋 Extracted Entities:")
            print("-" * 40)
            print(result.to_json())
            
            if not result.success:
                print(f"\n⚠️ Warning: {result.error or 'Could not parse JSON from response'}")
                print(f"Raw response: {result.raw_response[:200]}...")


if __name__ == "__main__":
    main()
