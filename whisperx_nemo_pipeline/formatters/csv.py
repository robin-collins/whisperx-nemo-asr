"""
CSV formatter for transcription output.
"""

import csv
import io
from typing import Dict, List, Any
from .base import BaseFormatter


class CSVFormatter(BaseFormatter):
    """
    Formats transcription data as CSV for data analysis workflows.
    Produces structured data with columns for speaker, text, timing, and metadata.
    """

    def __init__(self, include_word_level: bool = False):
        """
        Initialize the CSV formatter.
        
        Args:
            include_word_level: If True, export word-level data instead of sentence-level
        """
        self.include_word_level = include_word_level

    def format(self, transcript_data: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        Format transcript data as CSV.
        
        Args:
            transcript_data: List of sentence/segment dictionaries
            metadata: Optional metadata to include in the CSV
            
        Returns:
            CSV-formatted string
        """
        if not transcript_data:
            return ""
        
        # Create string buffer for CSV output
        output = io.StringIO()
        
        # Define CSV columns based on data structure
        fieldnames = [
            'segment_id',
            'speaker',
            'text',
            'start_time_ms',
            'end_time_ms',
            'start_time_seconds',
            'end_time_seconds',
            'duration_seconds'
        ]
        
        # Add metadata columns if available
        if metadata:
            fieldnames.extend(['language', 'backend', 'model'])
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data rows
        for i, segment in enumerate(transcript_data, start=1):
            start_ms = segment.get('start_time', 0)
            end_ms = segment.get('end_time', 0)
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            duration_sec = end_sec - start_sec
            
            row = {
                'segment_id': i,
                'speaker': segment.get('speaker', 'Unknown'),
                'text': segment.get('text', '').strip(),
                'start_time_ms': start_ms,
                'end_time_ms': end_ms,
                'start_time_seconds': round(start_sec, 3),
                'end_time_seconds': round(end_sec, 3),
                'duration_seconds': round(duration_sec, 3)
            }
            
            # Add metadata if available
            if metadata:
                row.update({
                    'language': metadata.get('language', ''),
                    'backend': metadata.get('backend', ''),
                    'model': metadata.get('model', '')
                })
            
            writer.writerow(row)
        
        return output.getvalue()

    def file_extension(self) -> str:
        """Return the file extension for CSV files."""
        return "csv"

    def format_name(self) -> str:
        """Return the human-readable name for this format."""
        return "CSV (Comma Separated Values)"


class WordLevelCSVFormatter(CSVFormatter):
    """
    CSV formatter for word-level transcription data.
    Requires access to word_timestamps data from TranscriptionResult.
    """
    
    def __init__(self):
        super().__init__(include_word_level=True)
    
    def format_word_timestamps(self, word_timestamps: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        Format word-level timestamps as CSV.
        
        Args:
            word_timestamps: List of word timestamp dictionaries
            metadata: Optional metadata
            
        Returns:
            CSV-formatted string with word-level data
        """
        if not word_timestamps:
            return ""
        
        output = io.StringIO()
        
        fieldnames = [
            'word_id',
            'word',
            'start_time_ms', 
            'end_time_ms',
            'start_time_seconds',
            'end_time_seconds',
            'confidence_score'
        ]
        
        if metadata:
            fieldnames.extend(['language', 'backend', 'model'])
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, word_data in enumerate(word_timestamps, start=1):
            start_sec = word_data.get('start', 0)
            end_sec = word_data.get('end', 0)
            start_ms = int(start_sec * 1000) if start_sec else 0
            end_ms = int(end_sec * 1000) if end_sec else 0
            
            row = {
                'word_id': i,
                'word': word_data.get('text', '').strip(),
                'start_time_ms': start_ms,
                'end_time_ms': end_ms, 
                'start_time_seconds': round(start_sec, 3) if start_sec else 0,
                'end_time_seconds': round(end_sec, 3) if end_sec else 0,
                'confidence_score': word_data.get('score', '')
            }
            
            if metadata:
                row.update({
                    'language': metadata.get('language', ''),
                    'backend': metadata.get('backend', ''),
                    'model': metadata.get('model', '')
                })
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def format_name(self) -> str:
        return "Word-level CSV"