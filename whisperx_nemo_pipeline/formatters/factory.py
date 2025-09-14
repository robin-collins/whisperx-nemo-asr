"""
Factory for creating output formatters.
"""

from typing import Dict, Type, List
from .base import BaseFormatter
from .srt import SRTFormatter
from .webvtt import WebVTTFormatter
from .plaintext import PlainTextFormatter, TimestampedTextFormatter, CleanTextFormatter
from .csv import CSVFormatter, WordLevelCSVFormatter


class FormatterFactory:
    """
    Factory class for creating output formatters.
    """
    
    _formatters: Dict[str, Type[BaseFormatter]] = {
        'srt': SRTFormatter,
        'webvtt': WebVTTFormatter,
        'vtt': WebVTTFormatter,  # Alias for webvtt
        'txt': PlainTextFormatter,
        'plaintext': PlainTextFormatter,
        'text': PlainTextFormatter,  # Alias for plaintext
        'timestamped': TimestampedTextFormatter,
        'clean': CleanTextFormatter,
        'csv': CSVFormatter,
        'wordcsv': WordLevelCSVFormatter,
    }

    @classmethod
    def create_formatter(cls, format_name: str) -> BaseFormatter:
        """
        Create a formatter instance for the specified format.
        
        Args:
            format_name: Name of the format (e.g., 'srt', 'vtt', 'txt')
            
        Returns:
            Formatter instance
            
        Raises:
            ValueError: If the format is not supported
        """
        format_name = format_name.lower()
        if format_name not in cls._formatters:
            available_formats = ', '.join(cls._formatters.keys())
            raise ValueError(f"Unsupported format '{format_name}'. Available formats: {available_formats}")
        
        formatter_class = cls._formatters[format_name]
        return formatter_class()

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get list of supported format names.
        
        Returns:
            List of supported format names
        """
        return list(cls._formatters.keys())

    @classmethod
    def register_formatter(cls, format_name: str, formatter_class: Type[BaseFormatter]) -> None:
        """
        Register a new formatter class.
        
        Args:
            format_name: Name of the format
            formatter_class: Formatter class to register
        """
        cls._formatters[format_name.lower()] = formatter_class