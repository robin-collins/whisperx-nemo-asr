"""
WebVTT (Web Video Text Tracks) formatter for transcription output.
"""

from typing import Dict, List, Any
from .base import BaseFormatter
from ..helpers import format_timestamp


class WebVTTFormatter(BaseFormatter):
    """
    Formats transcription data as WebVTT subtitles with speaker labels.
    WebVTT is the web standard format for subtitles, similar to SRT but with 
    different timestamp format (uses dots instead of commas for milliseconds).
    """

    def format(self, transcript_data: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        Format transcript data as WebVTT content.
        
        Args:
            transcript_data: List of sentence/segment dictionaries
            metadata: Optional metadata (unused for WebVTT format)
            
        Returns:
            WebVTT-formatted string
        """
        webvtt_lines = ["WEBVTT", ""]  # WebVTT header
        
        for i, segment in enumerate(transcript_data, start=1):
            # WebVTT uses dots instead of commas for milliseconds
            start_time = format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker='.')
            end_time = format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker='.')
            
            # Optional cue identifier
            webvtt_lines.append(f"{i}")
            
            # Timestamp line
            webvtt_lines.append(f"{start_time} --> {end_time}")
            
            # Subtitle text with speaker label
            webvtt_lines.append(f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}")
            
            # Empty line between cues
            webvtt_lines.append("")

        return "\n".join(webvtt_lines)

    def file_extension(self) -> str:
        """Return the file extension for WebVTT files."""
        return "vtt"

    def format_name(self) -> str:
        """Return the human-readable name for this format."""
        return "WebVTT"