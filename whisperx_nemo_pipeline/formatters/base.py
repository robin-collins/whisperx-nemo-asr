"""
Base formatter class for transcription output formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseFormatter(ABC):
    """
    Abstract base class for all output formatters.
    """

    @abstractmethod
    def format(self, transcript_data: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        Format the transcript data into the desired output format.
        
        Args:
            transcript_data: List of sentence/segment dictionaries containing:
                - speaker: Speaker identifier (e.g., "Speaker 0")
                - text: The transcribed text
                - start_time: Start timestamp in milliseconds
                - end_time: End timestamp in milliseconds
            metadata: Optional metadata about the transcription process
            
        Returns:
            Formatted string in the target format
        """
        pass

    @abstractmethod
    def file_extension(self) -> str:
        """
        Return the file extension for this format (without the dot).
        
        Returns:
            File extension string (e.g., 'srt', 'vtt', 'txt')
        """
        pass

    @abstractmethod
    def format_name(self) -> str:
        """
        Return the human-readable name of this format.
        
        Returns:
            Format name string (e.g., 'SubRip Subtitle', 'WebVTT', 'Plain Text')
        """
        pass