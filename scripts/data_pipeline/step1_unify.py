#!/usr/bin/env python3
"""
Step 1: Data Unification Script
================================

Reads various data formats (XML, JSON, CSV, MBOX) and
combines them into a single standardized DataFrame.

Output Schema: ['timestamp', 'sender', 'body', 'source']

Usage:
    python step1_unify.py --input /path/to/raw/data --output step1_unified.csv
"""

import argparse
import json
import csv
import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import mailbox
import email
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET


def parse_mbox(filepath: Path) -> List[Dict[str, Any]]:
    """Parse Gmail MBOX export."""
    records = []
    try:
        mbox = mailbox.mbox(str(filepath))
        for message in mbox:
            try:
                # Get timestamp
                date_str = message.get('Date', '')
                try:
                    timestamp = parsedate_to_datetime(date_str).isoformat()
                except:
                    timestamp = date_str
                
                # Get sender
                sender = message.get('From', '')
                
                # Get body
                body = ''
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == 'text/plain':
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode('utf-8', errors='ignore')
                                break
                else:
                    payload = message.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                
                if body.strip():
                    records.append({
                        'timestamp': timestamp,
                        'sender': sender,
                        'body': body.strip(),
                        'source': 'mbox'
                    })
            except Exception as e:
                continue
    except Exception as e:
        print(f"  ⚠️ Error parsing MBOX {filepath}: {e}")
    
    return records


def parse_json(filepath: Path) -> List[Dict[str, Any]]:
    """Parse JSON exports (Google Takeout format)."""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Common Google Takeout patterns
            items = data.get('messages', []) or \
                   data.get('transactions', []) or \
                   data.get('items', []) or \
                   data.get('data', []) or \
                   [data]
        else:
            items = []
        
        for item in items:
            if not isinstance(item, dict):
                continue
            
            # Try common field names
            timestamp = item.get('timestamp') or item.get('date') or \
                       item.get('time') or item.get('created_at') or ''
            
            sender = item.get('sender') or item.get('from') or \
                    item.get('source') or item.get('merchant') or ''
            
            body = item.get('body') or item.get('message') or \
                  item.get('text') or item.get('content') or \
                  item.get('description') or item.get('title') or ''
            
            # For Google Pay transactions
            if 'amount' in item:
                amount = item.get('amount', '')
                merchant = item.get('merchant', {})
                if isinstance(merchant, dict):
                    merchant_name = merchant.get('name', '')
                else:
                    merchant_name = str(merchant)
                body = f"Transaction: Rs.{amount} to {merchant_name}"
            
            if body and str(body).strip():
                records.append({
                    'timestamp': str(timestamp),
                    'sender': str(sender),
                    'body': str(body).strip(),
                    'source': f'json:{filepath.name}'
                })
    
    except Exception as e:
        print(f"  ⚠️ Error parsing JSON {filepath}: {e}")
    
    return records


def parse_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Parse CSV exports."""
    records = []
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        
        # Find relevant columns (case-insensitive)
        cols = {c.lower(): c for c in df.columns}
        
        timestamp_col = None
        for name in ['timestamp', 'date', 'time', 'datetime', 'created_at']:
            if name in cols:
                timestamp_col = cols[name]
                break
        
        sender_col = None
        for name in ['sender', 'from', 'source', 'bank', 'merchant']:
            if name in cols:
                sender_col = cols[name]
                break
        
        body_col = None
        for name in ['body', 'message', 'text', 'content', 'description', 'sms']:
            if name in cols:
                body_col = cols[name]
                break
        
        if body_col:
            for _, row in df.iterrows():
                body = str(row.get(body_col, ''))
                if body.strip() and body != 'nan':
                    records.append({
                        'timestamp': str(row.get(timestamp_col, '')) if timestamp_col else '',
                        'sender': str(row.get(sender_col, '')) if sender_col else '',
                        'body': body.strip(),
                        'source': f'csv:{filepath.name}'
                    })
    
    except Exception as e:
        print(f"  ⚠️ Error parsing CSV {filepath}: {e}")
    
    return records


def parse_xml(filepath: Path) -> List[Dict[str, Any]]:
    """Parse XML exports (SMS Backup format)."""
    records = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Common SMS backup format
        for sms in root.findall('.//sms') or root.findall('.//message'):
            body = sms.get('body') or sms.text or ''
            timestamp = sms.get('date') or sms.get('timestamp') or ''
            sender = sms.get('address') or sms.get('sender') or sms.get('from') or ''
            
            if body.strip():
                # Convert timestamp if it's milliseconds
                if timestamp.isdigit() and len(timestamp) > 10:
                    try:
                        timestamp = datetime.fromtimestamp(int(timestamp)/1000).isoformat()
                    except:
                        pass
                
                records.append({
                    'timestamp': timestamp,
                    'sender': sender,
                    'body': body.strip(),
                    'source': f'xml:{filepath.name}'
                })
    
    except Exception as e:
        print(f"  ⚠️ Error parsing XML {filepath}: {e}")
    
    return records


def find_all_files(input_dir: Path) -> Dict[str, List[Path]]:
    """Find all data files recursively."""
    files = {
        'mbox': [],
        'json': [],
        'csv': [],
        'xml': []
    }
    
    for filepath in input_dir.rglob('*'):
        if filepath.is_file():
            ext = filepath.suffix.lower()
            if ext == '.mbox':
                files['mbox'].append(filepath)
            elif ext == '.json':
                files['json'].append(filepath)
            elif ext == '.csv':
                files['csv'].append(filepath)
            elif ext == '.xml':
                files['xml'].append(filepath)
    
    return files


def unify_data(input_dir: Path) -> pd.DataFrame:
    """Main function to unify all data sources."""
    print("=" * 60)
    print("📂 STEP 1: DATA UNIFICATION")
    print("=" * 60)
    
    all_records = []
    
    # Find all files
    print(f"\n🔍 Scanning: {input_dir}")
    files = find_all_files(input_dir)
    
    total_files = sum(len(v) for v in files.values())
    print(f"   Found {total_files} files to process")
    
    # Parse MBOX files
    if files['mbox']:
        print(f"\n📧 Processing {len(files['mbox'])} MBOX files...")
        for f in files['mbox']:
            print(f"   Processing: {f.name}")
            records = parse_mbox(f)
            all_records.extend(records)
            print(f"   ✅ Extracted {len(records)} messages")
    
    # Parse JSON files
    if files['json']:
        print(f"\n📋 Processing {len(files['json'])} JSON files...")
        for f in files['json']:
            print(f"   Processing: {f.name}")
            records = parse_json(f)
            all_records.extend(records)
            print(f"   ✅ Extracted {len(records)} records")
    
    # Parse CSV files
    if files['csv']:
        print(f"\n📊 Processing {len(files['csv'])} CSV files...")
        for f in files['csv']:
            print(f"   Processing: {f.name}")
            records = parse_csv(f)
            all_records.extend(records)
            print(f"   ✅ Extracted {len(records)} records")
    
    # Parse XML files
    if files['xml']:
        print(f"\n📝 Processing {len(files['xml'])} XML files...")
        for f in files['xml']:
            print(f"   Processing: {f.name}")
            records = parse_xml(f)
            all_records.extend(records)
            print(f"   ✅ Extracted {len(records)} records")
    
    # Create DataFrame
    df = pd.DataFrame(all_records, columns=['timestamp', 'sender', 'body', 'source'])
    
    # Remove exact duplicates
    original_count = len(df)
    df = df.drop_duplicates(subset=['body'])
    dedup_count = len(df)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total records: {original_count}")
    print(f"   After dedup:   {dedup_count}")
    print(f"   Removed:       {original_count - dedup_count} duplicates")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Step 1: Unify data sources")
    parser.add_argument("--input", "-i", required=True, help="Input directory with raw data")
    parser.add_argument("--output", "-o", default="data/pipeline/step1_unified.csv", 
                       help="Output CSV path")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Unify data
    df = unify_data(input_dir)
    
    if len(df) == 0:
        print("\n❌ No data extracted! Check your input directory.")
        return
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Records: {len(df)}")
    print("\nNext: python scripts/data_pipeline/step2_filter.py")


if __name__ == "__main__":
    main()
