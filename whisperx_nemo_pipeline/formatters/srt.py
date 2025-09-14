"""
SRT (SubRip Subtitle) formatter for transcription output.
"""

from typing import Dict, List, Any
from .base import BaseFormatter
from ..helpers import format_timestamp


class SRTFormatter(BaseFormatter):
    """
    Formats transcription data as SRT subtitles with speaker labels.
    """

    def format(self, transcript_data: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        Format transcript data as SRT content.
        
        Args:
            transcript_data: List of sentence/segment dictionaries
            metadata: Optional metadata (unused for SRT format)
            
        Returns:
            SRT-formatted string
        """
        srt_lines = []
        for i, segment in enumerate(transcript_data, start=1):
            srt_lines.append(f"{i}")
            srt_lines.append(
                f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}"
            )
            srt_lines.append(
                f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}"
            )
            srt_lines.append("")  # Empty line between segments

        return "\n".join(srt_lines)

    def file_extension(self) -> str:
        """Return the file extension for SRT files."""
        return "srt"

    def format_name(self) -> str:
        """Return the human-readable name for this format."""
        return "SubRip Subtitle"