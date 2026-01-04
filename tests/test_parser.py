"""
Tests for the EmailParser class.
Run with: pytest tests/test_parser.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestEmailParserImport:
    """Test that EmailParser can be imported."""
    
    def test_import_parser(self):
        """Test that parser module can be imported."""
        try:
            from data.parser import EmailParser
            assert EmailParser is not None
        except ImportError as e:
            pytest.skip(f"Could not import EmailParser: {e}")


class TestEmailParserMethods:
    """Test EmailParser methods with mock data."""
    
    def test_clean_text_removes_urls(self):
        """Test URL removal in _clean_text."""
        from data.parser import EmailParser
        
        # Create a mock parser (won't actually load MBOX)
        class MockParser(EmailParser):
            def __init__(self):
                # Skip file check in init
                pass
        
        parser = MockParser()
        
        text = "Visit https://example.com for more info"
        result = parser._clean_text(text)
        assert "https://example.com" not in result
    
    def test_clean_text_normalizes_whitespace(self):
        """Test whitespace normalization."""
        from data.parser import EmailParser
        
        class MockParser(EmailParser):
            def __init__(self):
                pass
        
        parser = MockParser()
        
        text = "Hello    World\n\nTest"
        result = parser._clean_text(text)
        assert "  " not in result  # No double spaces
    
    def test_decode_header_none(self):
        """Test handling of None header."""
        from data.parser import EmailParser
        
        class MockParser(EmailParser):
            def __init__(self):
                pass
        
        parser = MockParser()
        
        result = parser._decode_header(None)
        assert result == ""
    
    def test_decode_header_plain_string(self):
        """Test decoding plain string header."""
        from data.parser import EmailParser
        
        class MockParser(EmailParser):
            def __init__(self):
                pass
        
        parser = MockParser()
        
        result = parser._decode_header("Simple Subject")
        assert result == "Simple Subject"


class TestEmailParserIntegration:
    """Integration tests that require actual files."""
    
    def test_parser_file_not_found(self):
        """Test that missing file raises error."""
        from data.parser import EmailParser
        
        with pytest.raises(FileNotFoundError):
            EmailParser(Path("/nonexistent/path/file.mbox"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
