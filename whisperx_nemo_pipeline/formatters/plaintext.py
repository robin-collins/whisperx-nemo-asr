"""
Plain Text formatter for transcription output.
"""

from typing import Dict, List, Any
from .base import BaseFormatter


class PlainTextFormatter(BaseFormatter):
    """
    Formats transcription data as plain text with speaker labels.
    Produces clean, readable transcripts without timing information.
    """

    def __init__(self, include_timestamps: bool = False, include_speaker_labels: bool = True):
        """
        Initialize the plain text formatter.
        
        Args:
            include_timestamps: Whether to include timestamp information
            include_speaker_labels: Whether to include speaker labels
        """
        self.include_timestamps = include_timestamps
        self.include_speaker_labels = include_speaker_labels

    def format(self, transcript_data: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        Format transcript data as plain text.
        
        Args:
            transcript_data: List of sentence/segment dictionaries
            metadata: Optional metadata (unused for plain text format)
            
        Returns:
            Plain text formatted string
        """
        if not transcript_data:
            return ""
        
        lines = []
        previous_speaker = None
        
        for segment in transcript_data:
            speaker = segment.get('speaker', 'Unknown Speaker')
            text = segment.get('text', '').strip()
            
            if not text:
                continue
            
            # Start new paragraph for speaker changes
            if self.include_speaker_labels and speaker != previous_speaker:
                if lines:  # Add blank line between speakers
                    lines.append("")
                
                if self.include_timestamps:
                    start_seconds = segment.get('start_time', 0) / 1000.0
                    lines.append(f"{speaker} ({start_seconds:.1f}s): {text}")
                else:
                    lines.append(f"{speaker}: {text}")
                    
                previous_speaker = speaker
            else:
                # Continue same speaker's text
                if self.include_speaker_labels:
                    lines.append(text)
                else:
                    lines.append(text)
        
        return "\n".join(lines)

    def file_extension(self) -> str:
        """Return the file extension for plain text files."""
        return "txt"

    def format_name(self) -> str:
        """Return the human-readable name for this format."""
        if self.include_timestamps:
            return "Plain Text with Timestamps"
        return "Plain Text"


class TimestampedTextFormatter(PlainTextFormatter):
    """
    Plain text formatter that includes timestamps.
    """
    
    def __init__(self):
        super().__init__(include_timestamps=True, include_speaker_labels=True)
        
    def format_name(self) -> str:
        return "Timestamped Text"


class CleanTextFormatter(PlainTextFormatter):
    """
    Plain text formatter without speaker labels or timestamps.
    Produces the cleanest possible transcript.
    """
    
    def __init__(self):
        super().__init__(include_timestamps=False, include_speaker_labels=False)
    
    def format(self, transcript_data: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """Format as clean text without any labels."""
        if not transcript_data:
            return ""
        
        text_segments = []
        for segment in transcript_data:
            text = segment.get('text', '').strip()
            if text:
                text_segments.append(text)
        
        return " ".join(text_segments)
    
    def format_name(self) -> str:
        return "Clean Text"