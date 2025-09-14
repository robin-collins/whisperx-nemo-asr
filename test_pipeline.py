#!/usr/bin/env python3
"""
Simple test script for the transcription pipeline.
"""

import json
import time
from pathlib import Path
from whisperx_nemo_pipeline.transcription_pipeline import create_transcription_pipeline


def save_transcription_results(result, audio_file, output_dir="output", keep_temp_files=False):
    """Save transcription results to files."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Extract audio filename without extension for output naming
    audio_name = Path(audio_file).stem

    # Save SRT content
    srt_filename = f"{audio_name}_transcription.srt"
    srt_filepath = output_path / srt_filename
    with open(srt_filepath, 'w', encoding='utf-8') as f:
        f.write(result.srt_content)
    print(f"SRT content saved to: {srt_filepath}")

    # Save word timestamps as JSON
    timestamps_filename = f"{audio_name}_word_timestamps.json"
    timestamps_filepath = output_path / timestamps_filename
    with open(timestamps_filepath, 'w', encoding='utf-8') as f:
        json.dump(result.word_timestamps, f, indent=2, ensure_ascii=False)
    print(f"Word timestamps saved to: {timestamps_filepath}")

    # Save metadata
    metadata_filename = f"{audio_name}_metadata.json"
    metadata_filepath = output_path / metadata_filename
    metadata = {
        "audio_file": audio_file,
        "language": result.language,
        "backend": result.backend,
        "total_time_seconds": result.total_time,
        "number_of_words": len(result.word_timestamps),
        "processing_timestamp": time.time(),
        "model": "base",
        "batch_size": 4,
        "stemming": False,
        "temp_files_kept": keep_temp_files
    }
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {metadata_filepath}")

    return {
        "srt_file": str(srt_filepath),
        "timestamps_file": str(timestamps_filepath),
        "metadata_file": str(metadata_filepath)
    }


def test_pipeline(save_output=True, keep_temp_files=False):
    """Test the transcription pipeline with WhisperX backend."""

    audio_file = "audio/two-speaker-dialog.wav"

    print("Testing transcription pipeline with:")
    print(f"  Audio file: {audio_file}")
    print("  Backend: whisperx")
    print("  Model: base")
    print("  Batch size: 4")
    print(f"  Keep temp files: {keep_temp_files}")
    print("-" * 50)

    # Create pipeline
    pipeline = create_transcription_pipeline(
        audio_path=audio_file,
        model_name="base",
        batch_size=4,
        backend="whisperx",
        stemming=False,  # Disable source separation to avoid demucs issues
        keep_temp_files=keep_temp_files,
    )

    # Process audio
    start_time = time.time()
    result = pipeline.process()
    processing_time = time.time() - start_time

    # Display results
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Detected language: {result.language}")
    print(f"Backend used: {result.backend}")
    print(f"Total pipeline time: {result.total_time:.2f} seconds")
    print(f"Number of word timestamps: {len(result.word_timestamps)}")
    print("-" * 50)

    # Show first few word timestamps
    print("First 5 word timestamps:")
    for i, word in enumerate(result.word_timestamps[:5]):
        print(f"  {i + 1}: {word}")

    print("-" * 50)

    # Show beginning of SRT content
    srt_lines = result.srt_content.split("\n")
    print("First 10 lines of SRT content:")
    for i, line in enumerate(srt_lines[:10]):
        print(f"  {line}")

    print("-" * 50)

    # Save results if requested
    if save_output:
        print("Saving transcription results...")
        saved_files = save_transcription_results(result, audio_file, keep_temp_files=keep_temp_files)
        print("-" * 50)
        print("Files saved:")
        for file_type, filepath in saved_files.items():
            print(f"  {file_type}: {filepath}")
        print("-" * 50)

    print("Test completed successfully!")

    return result


if __name__ == "__main__":
    import sys

    # Check for command line arguments
    save_output = "--no-save" not in sys.argv
    keep_temp_files = "--keep-temp" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python test_pipeline.py [--no-save] [--keep-temp] [--help]")
        print("")
        print("Options:")
        print("  --no-save       Don't save transcription results to files")
        print("  --keep-temp     Keep temporary files after processing")
        print("  --help, -h      Show this help message")
        print("")
        print("Examples:")
        print("  python test_pipeline.py                    # Save output, delete temp files")
        print("  python test_pipeline.py --keep-temp       # Save output, keep temp files")
        print("  python test_pipeline.py --no-save         # Don't save, delete temp files")
        print("  python test_pipeline.py --keep-temp --no-save  # Don't save, keep temp files")
        sys.exit(0)

    result = test_pipeline(save_output=save_output, keep_temp_files=keep_temp_files)
