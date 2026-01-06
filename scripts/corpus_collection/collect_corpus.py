"""
Domain Pre-training Corpus Collector.

Collects and prepares text corpus for domain pre-training on Indian
financial domain. Combines multiple sources into a unified JSONL format.

Sources:
    1. Gmail bank emails (MBOX export)
    2. Bank statement PDFs (text extraction)
    3. RBI circulars and guidelines
    4. NPCI/UPI documentation
    5. Financial news articles
    6. Banking glossary

Target: 10M+ tokens for effective domain pre-training.

Author: Ranjit Behera
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CorpusDocument:
    """A single document in the corpus."""
    text: str
    source: str  # email, statement, rbi, npci, news, glossary
    metadata: Dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.text.split())
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "word_count": self.word_count
        }


@dataclass
class CorpusStats:
    """Statistics for the corpus."""
    total_documents: int = 0
    total_words: int = 0
    total_chars: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    
    def estimated_tokens(self) -> int:
        """Estimate tokens (roughly 0.75 tokens per word for English)."""
        return int(self.total_words * 1.3)  # Financial text has more tokens


class CorpusCollector:
    """
    Collects and manages the domain pre-training corpus.
    
    Usage:
        collector = CorpusCollector()
        collector.add_glossary()
        collector.add_emails_from_mbox("path/to/mbox")
        collector.add_statements_from_pdfs("path/to/pdfs")
        collector.export("data/corpus/combined/corpus.jsonl")
    """
    
    def __init__(self, corpus_dir: str = "data/corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents: List[CorpusDocument] = []
        self.stats = CorpusStats()
    
    def add_document(self, text: str, source: str, metadata: Dict = None):
        """Add a single document to the corpus."""
        # Clean text
        text = self._clean_text(text)
        
        if len(text.split()) < 10:  # Skip very short texts
            return
        
        doc = CorpusDocument(
            text=text,
            source=source,
            metadata=metadata or {}
        )
        
        self.documents.append(doc)
        self.stats.total_documents += 1
        self.stats.total_words += doc.word_count
        self.stats.total_chars += len(text)
        self.stats.by_source[source] = self.stats.by_source.get(source, 0) + 1
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s₹$€£¥.,;:!?@#%&*()[\]{}<>+=\-/\\|\'\"°\n]', '', text)
        # Normalize newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def add_glossary(self):
        """Add banking glossary to corpus."""
        from src.data.banking_glossary import get_full_glossary
        
        glossary = get_full_glossary()
        
        # Split into sections
        sections = glossary.split('\n\n')
        for section in sections:
            if len(section.strip()) > 50:
                self.add_document(
                    text=section.strip(),
                    source="glossary",
                    metadata={"type": "banking_glossary"}
                )
        
        logger.info(f"Added glossary: {self.stats.by_source.get('glossary', 0)} sections")
    
    def add_emails_from_mbox(self, mbox_path: str):
        """Extract and add emails from MBOX file."""
        import mailbox
        from bs4 import BeautifulSoup
        
        mbox_path = Path(mbox_path)
        if not mbox_path.exists():
            logger.warning(f"MBOX file not found: {mbox_path}")
            return
        
        logger.info(f"Processing MBOX: {mbox_path}")
        
        mbox = mailbox.mbox(str(mbox_path))
        count = 0
        
        for message in mbox:
            try:
                # Get sender
                sender = message.get('From', '')
                subject = message.get('Subject', '')
                
                # Filter for bank-related emails
                if not self._is_financial_email(sender, subject):
                    continue
                
                # Extract body
                body = self._extract_email_body(message)
                if not body:
                    continue
                
                self.add_document(
                    text=f"Subject: {subject}\n\n{body}",
                    source="email",
                    metadata={
                        "sender": sender[:100],
                        "subject": subject[:200],
                        "date": message.get('Date', '')
                    }
                )
                count += 1
                
            except Exception as e:
                logger.debug(f"Error processing email: {e}")
        
        logger.info(f"Added {count} financial emails")
    
    def _is_financial_email(self, sender: str, subject: str) -> bool:
        """Check if email is finance-related."""
        keywords = [
            'bank', 'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yes bank',
            'paytm', 'phonepe', 'gpay', 'google pay', 'amazon pay',
            'mutual fund', 'investment', 'credit', 'debit', 'transaction',
            'upi', 'neft', 'rtgs', 'imps', 'payment', 'transfer',
            'loan', 'emi', 'insurance', 'lic', 'premium',
            'zerodha', 'groww', 'upstox', 'cams', 'karvy',
            'income tax', 'gst', 'tds', 'itr', 'form 16'
        ]
        
        text = f"{sender} {subject}".lower()
        return any(kw in text for kw in keywords)
    
    def _extract_email_body(self, message) -> Optional[str]:
        """Extract text body from email message."""
        from bs4 import BeautifulSoup
        
        body = ""
        
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except:
                        pass
                elif content_type == 'text/html':
                    try:
                        html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(html, 'html.parser')
                        body = soup.get_text(separator=' ', strip=True)
                    except:
                        pass
        else:
            try:
                content_type = message.get_content_type()
                payload = message.get_payload(decode=True)
                if payload:
                    text = payload.decode('utf-8', errors='ignore')
                    if content_type == 'text/html':
                        soup = BeautifulSoup(text, 'html.parser')
                        body = soup.get_text(separator=' ', strip=True)
                    else:
                        body = text
            except:
                pass
        
        return body.strip() if body else None
    
    def add_statements_from_pdfs(self, pdf_dir: str):
        """Extract text from bank statement PDFs."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed. Skipping PDF extraction.")
            return
        
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return
        
        pdf_files = list(pdf_dir.glob("**/*.pdf")) + list(pdf_dir.glob("**/*.PDF"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n\n"
                    
                    if full_text.strip():
                        self.add_document(
                            text=full_text,
                            source="statement",
                            metadata={
                                "filename": pdf_path.name,
                                "pages": len(pdf.pages)
                            }
                        )
                        
            except Exception as e:
                logger.debug(f"Error processing {pdf_path.name}: {e}")
        
        logger.info(f"Added {self.stats.by_source.get('statement', 0)} statements")
    
    def add_text_files(self, dir_path: str, source: str):
        """Add all text files from a directory."""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return
        
        for txt_file in dir_path.glob("**/*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                self.add_document(
                    text=text,
                    source=source,
                    metadata={"filename": txt_file.name}
                )
            except Exception as e:
                logger.debug(f"Error reading {txt_file}: {e}")
        
        logger.info(f"Added {self.stats.by_source.get(source, 0)} documents from {source}")
    
    def add_custom_text(self, text: str, source: str = "custom"):
        """Add custom text to corpus."""
        self.add_document(text=text, source=source)
    
    def export(self, output_path: str = None) -> Path:
        """Export corpus to JSONL file."""
        if output_path is None:
            output_path = self.corpus_dir / "combined" / "corpus.jsonl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(json.dumps({"text": doc.text}) + '\n')
        
        logger.info(f"Exported corpus to {output_path}")
        return output_path
    
    def export_detailed(self, output_path: str = None) -> Path:
        """Export corpus with metadata to JSONL."""
        if output_path is None:
            output_path = self.corpus_dir / "combined" / "corpus_detailed.jsonl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(json.dumps(doc.to_dict()) + '\n')
        
        return output_path
    
    def print_stats(self):
        """Print corpus statistics."""
        print("\n" + "=" * 60)
        print("📊 CORPUS STATISTICS")
        print("=" * 60)
        print(f"Total Documents: {self.stats.total_documents:,}")
        print(f"Total Words:     {self.stats.total_words:,}")
        print(f"Est. Tokens:     {self.stats.estimated_tokens():,}")
        print(f"Total Chars:     {self.stats.total_chars:,}")
        print("\nBy Source:")
        for source, count in sorted(self.stats.by_source.items()):
            print(f"  {source:15} {count:,} documents")
        print("=" * 60)
        
        # Check if sufficient for pre-training
        tokens = self.stats.estimated_tokens()
        if tokens >= 10_000_000:
            print("✅ Sufficient for domain pre-training (10M+ tokens)")
        elif tokens >= 1_000_000:
            print("⚠️  Marginal corpus size. Consider adding more data.")
        else:
            print("❌ Corpus too small. Need at least 1M tokens, ideally 10M+")
    
    def analyze_vocabulary(self, top_n: int = 50) -> Dict[str, int]:
        """Analyze most common financial terms in corpus."""
        all_text = " ".join(doc.text for doc in self.documents)
        words = re.findall(r'\b[A-Za-z]{2,}\b', all_text.lower())
        
        # Focus on financial terms
        financial_words = []
        financial_keywords = {
            'upi', 'neft', 'rtgs', 'imps', 'bank', 'account', 'transaction',
            'payment', 'credit', 'debit', 'balance', 'transfer', 'amount',
            'rupees', 'rs', 'inr', 'loan', 'emi', 'interest', 'deposit',
            'withdrawal', 'statement', 'passbook', 'cheque', 'card'
        }
        
        for word in words:
            if word in financial_keywords or any(kw in word for kw in financial_keywords):
                financial_words.append(word)
        
        return dict(Counter(financial_words).most_common(top_n))


def main():
    """Main function to collect corpus."""
    print("=" * 60)
    print("🏦 Domain Pre-training Corpus Collector")
    print("=" * 60)
    
    collector = CorpusCollector()
    
    # 1. Add glossary (always available)
    print("\n📚 Adding banking glossary...")
    collector.add_glossary()
    
    # 2. Check for MBOX file
    mbox_paths = [
        Path("data/raw/All mail Including Spam and Trash.mbox"),
        Path("data/raw/emails.mbox"),
        Path.home() / "Downloads" / "Takeout" / "Mail" / "All mail Including Spam and Trash.mbox"
    ]
    
    for mbox_path in mbox_paths:
        if mbox_path.exists():
            print(f"\n📧 Processing emails from {mbox_path}...")
            collector.add_emails_from_mbox(str(mbox_path))
            break
    else:
        print("\n⚠️  No MBOX file found. Export Gmail via Google Takeout.")
    
    # 3. Check for PDFs
    pdf_dirs = [
        Path("data/raw/pdfs"),
        Path("data/raw/statements"),
    ]
    
    for pdf_dir in pdf_dirs:
        if pdf_dir.exists():
            print(f"\n📄 Extracting text from PDFs in {pdf_dir}...")
            collector.add_statements_from_pdfs(str(pdf_dir))
    
    # 4. Check for additional text files
    additional_dirs = [
        ("data/corpus/rbi", "rbi"),
        ("data/corpus/npci", "npci"),
        ("data/corpus/news", "news"),
    ]
    
    for dir_path, source in additional_dirs:
        if Path(dir_path).exists():
            print(f"\n📁 Adding {source} documents...")
            collector.add_text_files(dir_path, source)
    
    # Print stats
    collector.print_stats()
    
    # Export if we have data
    if collector.documents:
        output = collector.export()
        print(f"\n💾 Corpus saved to: {output}")
        
        # Also save detailed version
        collector.export_detailed()
    
    return collector


if __name__ == "__main__":
    main()
