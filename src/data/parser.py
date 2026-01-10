"""
Email Parser Module
Extracts and cleans emails from MBOX file locally.
"""

import mailbox
import email
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Generator
from bs4 import BeautifulSoup
from tqdm import tqdm


class EmailParser:
    """Parse emails from MBOX file."""
    
    def __init__(self, mbox_path: Path):
        self.mbox_path = Path(mbox_path)
        if not self.mbox_path.exists():
            raise FileNotFoundError(f"MBOX not found: {mbox_path}")
    
    def _decode_payload(self, message) -> str:
        """Extract text content from email."""
        try:
            if message.is_multipart():
                for part in message.walk():
                    ctype = part.get_content_type()
                    if ctype == 'text/plain':
                        payload = part.get_payload(decode=True)
                        if payload:
                            return payload.decode('utf-8', errors='ignore')
                    elif ctype == 'text/html':
                        payload = part.get_payload(decode=True)
                        if payload:
                            soup = BeautifulSoup(
                                payload.decode('utf-8', errors='ignore'), 
                                'lxml'
                            )
                            return soup.get_text(separator=' ', strip=True)
            else:
                payload = message.get_payload(decode=True)
                if payload:
                    return payload.decode('utf-8', errors='ignore')
        except Exception:
            pass
        return ''
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove long encoded strings
        text = re.sub(r'\S{100,}', '', text)
        return text.strip()
    
    def _decode_header(self, header) -> str:
        """Decode email header."""
        if header is None:
            return ''
        try:
            decoded = email.header.decode_header(header)
            parts = []
            for content, charset in decoded:
                if isinstance(content, bytes):
                    content = content.decode(charset or 'utf-8', errors='ignore')
                parts.append(str(content))
            return ' '.join(parts)
        except Exception:
            return str(header)
    
    def parse(
        self, 
        limit: Optional[int] = None,
        min_length: int = 50,
        max_length: int = 5000
    ) -> Generator[Dict, None, None]:
        """
        Parse emails from MBOX file.
        
        Yields email dictionaries one at a time (memory efficient).
        """
        mbox = mailbox.mbox(str(self.mbox_path))
        total = len(mbox) if limit is None else min(limit, len(mbox))
        
        print(f"Parsing {total:,} emails from {self.mbox_path.name}")
        
        count = 0
        for i, message in enumerate(tqdm(mbox, total=total, desc="Parsing")):
            if limit and i >= limit:
                break
            
            try:
                body = self._decode_payload(message)
                body = self._clean_text(body)
                
                # Skip if too short or empty
                if len(body) < min_length:
                    continue
                
                # Truncate if too long
                body = body[:max_length]
                
                yield {
                    'id': count,
                    'subject': self._clean_text(self._decode_header(message['subject'])),
                    'sender': self._clean_text(self._decode_header(message['from'])),
                    'date': message['date'] or '',
                    'body': body
                }
                count += 1
                
            except Exception as e:
                continue
        
        print(f"Successfully parsed {count:,} emails")
    
    def parse_and_save(
        self,
        output_path: Path,
        limit: Optional[int] = None,
        min_length: int = 50,
        max_length: int = 5000
    ) -> int:
        """Parse emails and save to JSON."""
        emails = list(self.parse(limit, min_length, max_length))
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emails, f, ensure_ascii=False, indent=2)
        
        print(f"Saved to {output_path}")
        return len(emails)


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Parse emails
    mbox_path = Path(config['paths']['raw_data']) / config['data']['mbox_file']
    output_path = Path(config['paths']['parsed_data']) / "emails.json"
    
    parser = EmailParser(mbox_path)
    count = parser.parse_and_save(
        output_path,
        limit=config['data']['max_emails'],
        min_length=config['data']['min_body_length'],
        max_length=config['data']['max_body_length']
    )
    
    print(f"\nTotal emails parsed: {count:,}")
