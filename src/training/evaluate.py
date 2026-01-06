"""
Model Evaluation Module for Phase 1 Completion.

Evaluates the fine-tuned model on the held-out test set,
computing per-bank accuracy and generating a completion report.

Author: Ranjit Behera
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime


@dataclass
class EvaluationResult:
    """Result for a single evaluation sample."""
    bank: str
    expected: Dict
    predicted: Dict
    raw_response: str
    correct_fields: Dict[str, bool] = field(default_factory=dict)
    overall_correct: bool = False
    inference_time_ms: float = 0.0
    
    def compute_accuracy(self):
        """Compute per-field accuracy."""
        for key in ["amount", "type", "date", "reference", "merchant", "category"]:
            expected_val = str(self.expected.get(key, "")).lower().strip()
            predicted_val = str(self.predicted.get(key, "")).lower().strip()
            
            # Normalize amount (remove commas, trailing zeros)
            if key == "amount":
                try:
                    expected_val = str(float(expected_val.replace(",", "")))
                    predicted_val = str(float(predicted_val.replace(",", "")))
                except:
                    pass
            
            self.correct_fields[key] = expected_val == predicted_val
        
        # Core fields that must match
        core_fields = ["amount", "type"]
        self.overall_correct = all(
            self.correct_fields.get(f, False) for f in core_fields
        )


@dataclass 
class BankMetrics:
    """Metrics for a single bank."""
    bank: str
    total: int = 0
    correct: int = 0
    field_correct: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    field_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_inference_time_ms: float = 0.0
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0
    
    def field_accuracy(self, field: str) -> float:
        total = self.field_total[field]
        return self.field_correct[field] / total if total > 0 else 0.0


class ModelEvaluator:
    """
    Evaluates the fine-tuned model on test data.
    
    Generates per-bank accuracy metrics and a phase completion report.
    """
    
    def __init__(
        self,
        model_path: str = "models/base/phi3-mini",
        adapter_path: str = "models/adapters/finance-lora-v3",
        test_file: str = "data/synthetic/test_emails.json",
        project_root: Optional[Path] = None
    ):
        self.project_root = project_root or Path.cwd()
        self.model_path = self.project_root / model_path
        self.adapter_path = self.project_root / adapter_path
        self.test_file = self.project_root / test_file
        
        self.predictor = None
        self.results: List[EvaluationResult] = []
        self.bank_metrics: Dict[str, BankMetrics] = {}
        
    def load_model(self):
        """Load the model with adapter."""
        from src.inference.predict import Predictor
        
        print(f"🔄 Loading model from {self.model_path}")
        print(f"   With adapter: {self.adapter_path}")
        
        self.predictor = Predictor(
            model_path=self.model_path,
            adapter_path=self.adapter_path,
            max_tokens=250
        )
        self.predictor.load()
        
    def load_test_data(self) -> List[Dict]:
        """Load test samples."""
        with open(self.test_file) as f:
            data = json.load(f)
        print(f"📋 Loaded {len(data)} test samples")
        return data
    
    def evaluate(self, limit: Optional[int] = None) -> Dict:
        """
        Run evaluation on the test set.
        
        Args:
            limit: Optional limit on number of samples to evaluate
            
        Returns:
            Summary metrics dictionary
        """
        if self.predictor is None:
            self.load_model()
        
        test_data = self.load_test_data()
        if limit:
            test_data = test_data[:limit]
        
        print(f"\n🧪 Running evaluation on {len(test_data)} samples...")
        print("=" * 60)
        
        for i, sample in enumerate(test_data):
            bank = sample.get("bank", "unknown")
            subject = sample.get("subject", "")
            body = sample.get("body", sample.get("raw_text", ""))
            expected = sample.get("entities", {})
            
            # Run inference
            start_time = time.time()
            result = self.predictor.predict(subject=subject, body=body)
            inference_time = (time.time() - start_time) * 1000
            
            # Create evaluation result
            eval_result = EvaluationResult(
                bank=bank,
                expected=expected,
                predicted=result.entities,
                raw_response=result.raw_response,
                inference_time_ms=inference_time
            )
            eval_result.compute_accuracy()
            self.results.append(eval_result)
            
            # Progress indicator
            status = "✓" if eval_result.overall_correct else "✗"
            if (i + 1) % 10 == 0 or i == len(test_data) - 1:
                print(f"   Processed {i + 1}/{len(test_data)} samples")
        
        # Compute bank-level metrics
        self._compute_bank_metrics()
        
        return self.get_summary()
    
    def _compute_bank_metrics(self):
        """Compute per-bank metrics."""
        bank_results = defaultdict(list)
        
        for result in self.results:
            bank_results[result.bank].append(result)
        
        for bank, results in bank_results.items():
            metrics = BankMetrics(bank=bank)
            metrics.total = len(results)
            metrics.correct = sum(1 for r in results if r.overall_correct)
            
            total_time = 0
            for result in results:
                total_time += result.inference_time_ms
                for field, correct in result.correct_fields.items():
                    metrics.field_total[field] += 1
                    if correct:
                        metrics.field_correct[field] += 1
            
            metrics.avg_inference_time_ms = total_time / len(results)
            self.bank_metrics[bank] = metrics
    
    def get_summary(self) -> Dict:
        """Get evaluation summary."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r.overall_correct)
        
        # Field-level accuracy
        field_accuracy = {}
        for field in ["amount", "type", "date", "reference", "merchant", "category"]:
            field_correct = sum(
                1 for r in self.results 
                if r.correct_fields.get(field, False)
            )
            field_accuracy[field] = field_correct / total if total > 0 else 0.0
        
        return {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / total if total > 0 else 0.0,
            "field_accuracy": field_accuracy,
            "per_bank": {
                bank: {
                    "total": m.total,
                    "correct": m.correct,
                    "accuracy": m.accuracy,
                    "avg_inference_ms": m.avg_inference_time_ms
                }
                for bank, m in self.bank_metrics.items()
            }
        }
    
    def print_report(self):
        """Print a formatted evaluation report."""
        summary = self.get_summary()
        
        print("\n")
        print("╔" + "═" * 62 + "╗")
        print("║" + " 📊 PHASE 1 EVALUATION REPORT".center(62) + "║")
        print("╠" + "═" * 62 + "╣")
        
        # Overall metrics
        acc_pct = summary["overall_accuracy"] * 100
        print(f"║  Overall Accuracy: {acc_pct:>6.1f}% ({summary['correct_samples']}/{summary['total_samples']})".ljust(63) + "║")
        print("╠" + "═" * 62 + "╣")
        
        # Per-bank accuracy
        print("║" + " 🏦 Per-Bank Accuracy".center(62) + "║")
        print("╠" + "─" * 62 + "╣")
        print("║  " + f"{'Bank':<12} {'Samples':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Time':>12}" + "   ║")
        print("╠" + "─" * 62 + "╣")
        
        for bank in sorted(summary["per_bank"].keys()):
            data = summary["per_bank"][bank]
            acc = data["accuracy"] * 100
            time_ms = data["avg_inference_ms"]
            emoji = "✅" if acc >= 80 else "⚠️" if acc >= 60 else "❌"
            print(f"║  {emoji} {bank.upper():<10} {data['total']:>6} {data['correct']:>8} {acc:>9.1f}% {time_ms:>10.0f}ms" + " ║")
        
        print("╠" + "═" * 62 + "╣")
        
        # Per-field accuracy
        print("║" + " 🎯 Per-Field Accuracy".center(62) + "║")
        print("╠" + "─" * 62 + "╣")
        
        for field, acc in sorted(summary["field_accuracy"].items(), key=lambda x: -x[1]):
            acc_pct = acc * 100
            bar_len = int(acc * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            emoji = "✅" if acc >= 0.8 else "⚠️" if acc >= 0.6 else "❌"
            print(f"║  {emoji} {field:<12} {bar} {acc_pct:>5.1f}%" + "   ║")
        
        print("╚" + "═" * 62 + "╝")
        
        # Phase 1 completion status
        print("\n")
        overall_acc = summary["overall_accuracy"]
        if overall_acc >= 0.85:
            print("🎉 PHASE 1 COMPLETE! Model achieves >85% accuracy on all banks.")
        elif overall_acc >= 0.75:
            print("✅ Phase 1 nearly complete. Consider more training for edge cases.")
        else:
            print("⚠️  Phase 1 needs more work. Review failed samples and retrain.")
    
    def save_report(self, output_path: Optional[Path] = None):
        """Save evaluation report to JSON."""
        if output_path is None:
            output_path = self.project_root / "data/evaluation_report.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "adapter_path": str(self.adapter_path),
            "test_file": str(self.test_file),
            "summary": self.get_summary(),
            "detailed_results": [
                {
                    "bank": r.bank,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "correct_fields": r.correct_fields,
                    "overall_correct": r.overall_correct,
                    "inference_time_ms": r.inference_time_ms
                }
                for r in self.results
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to {output_path}")
    
    def get_failed_samples(self) -> List[EvaluationResult]:
        """Get samples where prediction was incorrect."""
        return [r for r in self.results if not r.overall_correct]


def main():
    """Run evaluation and generate report."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", default="models/base/phi3-mini")
    parser.add_argument("--adapter", default="models/adapters/finance-lora-v3")
    parser.add_argument("--test-file", default="data/synthetic/test_emails.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--save", action="store_true", help="Save report to JSON")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        model_path=args.model,
        adapter_path=args.adapter,
        test_file=args.test_file
    )
    
    evaluator.evaluate(limit=args.limit)
    evaluator.print_report()
    
    if args.save:
        evaluator.save_report()


if __name__ == "__main__":
    main()
