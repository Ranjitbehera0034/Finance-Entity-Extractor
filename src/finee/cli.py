"""
FinEE CLI - Command-line interface for financial entity extraction.

Usage:
    finee extract "Rs.500 debited from A/c 1234 on 01-01-25"
    finee extract --file transactions.txt
    finee stats
    finee backends
"""

import argparse
import json
import sys
from typing import Optional
import logging

from .extractor import FinEE, extract, get_extractor
from .schema import ExtractionConfig
from .backends import get_available_backends


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def cmd_extract(args):
    """Handle extract command."""
    # Get text from argument or file
    if args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.text]
    
    # Configure extractor
    config = ExtractionConfig(
        use_llm=not args.no_llm,
        cache_enabled=not args.no_cache,
    )
    
    extractor = FinEE(config)
    
    # Extract
    for text in texts:
        result = extractor.extract(text)
        
        if args.json:
            print(result.to_json())
        else:
            print(f"\n{'='*60}")
            print(f"Input: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"{'='*60}")
            
            # Core fields
            print(f"Amount:     {result.amount}")
            print(f"Type:       {result.type.value if result.type else 'N/A'}")
            print(f"Date:       {result.date or 'N/A'}")
            print(f"Account:    {result.account or 'N/A'}")
            print(f"Reference:  {result.reference or 'N/A'}")
            
            # Enrichment
            print(f"Merchant:   {result.merchant or 'N/A'}")
            print(f"Category:   {result.category.value if result.category else 'N/A'}")
            
            # Metadata
            print(f"\nConfidence: {result.confidence.value} ({result.confidence_score:.0%})")
            print(f"Time:       {result.processing_time_ms:.2f}ms")
            print(f"Cached:     {result.from_cache}")


def cmd_stats(args):
    """Handle stats command."""
    extractor = get_extractor()
    stats = extractor.get_stats()
    
    print("\nFinEE Statistics")
    print("="*40)
    print(json.dumps(stats, indent=2))


def cmd_backends(args):
    """Handle backends command."""
    backends = get_available_backends()
    
    print("\nAvailable Backends")
    print("="*40)
    
    if backends:
        for backend in backends:
            print(f"  ✅ {backend}")
    else:
        print("  ⚠️  No LLM backends available")
        print("\nInstall a backend:")
        print("  pip install finee[metal]   # Apple Silicon")
        print("  pip install finee[cuda]    # NVIDIA GPU")
        print("  pip install finee[cpu]     # CPU (llama.cpp)")


def cmd_version(args):
    """Handle version command."""
    from . import __version__
    print(f"finee {__version__}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='finee',
        description='Extract structured financial entities from Indian banking messages'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='store_true', help='Show version')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract entities from text')
    extract_parser.add_argument('text', nargs='?', help='Transaction text')
    extract_parser.add_argument('-f', '--file', help='Read from file (one per line)')
    extract_parser.add_argument('--json', action='store_true', help='Output as JSON')
    extract_parser.add_argument('--no-llm', action='store_true', help='Disable LLM (regex only)')
    extract_parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    extract_parser.set_defaults(func=cmd_extract)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show extraction statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Backends command
    backends_parser = subparsers.add_parser('backends', help='List available backends')
    backends_parser.set_defaults(func=cmd_backends)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle version
    if args.version:
        cmd_version(args)
        return
    
    # Handle commands
    if hasattr(args, 'func'):
        # Validate extract command
        if args.command == 'extract':
            if not args.text and not args.file:
                extract_parser.error("Either TEXT or --file is required")
        
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
