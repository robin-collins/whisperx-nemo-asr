#!/usr/bin/env python3
"""
Simple test script for the transcription pipeline.
"""

import time
from whisperx_nemo_pipeline.transcription_pipeline import create_transcription_pipeline

def test_pipeline():
    """Test the transcription pipeline with WhisperX backend."""
    
    audio_file = "normalized_sales_call.wav"
    
    print(f"Testing transcription pipeline with:")
    print(f"  Audio file: {audio_file}")
    print(f"  Backend: whisperx")
    print(f"  Model: large-v3-turbo")
    print(f"  Batch size: 12")
    print("-" * 50)
    
    # Create pipeline
    pipeline = create_transcription_pipeline(
        audio_path=audio_file,
        model_name="large-v3-turbo",
        batch_size=12,
        backend="whisperx"
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
        print(f"  {i+1}: {word}")
    
    print("-" * 50)
    
    # Show beginning of SRT content
    srt_lines = result.srt_content.split('\n')
    print("First 10 lines of SRT content:")
    for i, line in enumerate(srt_lines[:10]):
        print(f"  {line}")
    
    print("-" * 50)
    print("Test completed successfully!")
    
    return result

if __name__ == "__main__":
    result = test_pipeline()