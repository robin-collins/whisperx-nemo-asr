# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WhisperX-NeMo Pipeline is a production-ready transcription and diarization pipeline that combines:
- **Transcription**: WhisperX or Faster-Whisper for speech-to-text
- **Diarization**: NeMo MSDD models for speaker identification
- **Parallel Processing**: Simultaneous transcription and diarization for efficiency
- **Audio Processing**: Optional Demucs vocal separation and punctuation restoration

## Core Architecture

### Main Components
- `whisperx_nemo_pipeline/transcription_pipeline.py`: Core pipeline logic with `TranscriptionPipeline` class and factory function `create_transcription_pipeline()`
- `whisperx_nemo_pipeline/helpers.py`: Utility functions for audio processing, configuration, and text alignment
- `whisperx_nemo_pipeline/__init__.py`: Main exports and entry points

### Key Classes
- `TranscriptionPipeline`: Main orchestrator that handles parallel processing
- `TranscriptionConfig`: Configuration dataclass for pipeline parameters
- `TranscriptionResult`/`DiarizationResult`: Result containers with timing and metadata

### Pipeline Flow
1. Audio preprocessing (optional Demucs source separation)
2. Parallel execution of transcription (Whisper) and diarization (NeMo)
3. Alignment of transcription with speaker timestamps
4. Optional punctuation restoration
5. SRT subtitle generation

## Development Commands

### Package Management (UV-based)
```bash
# Install dependencies
uv sync

# Install with development dependencies
uv sync --dev

# Install with GPU support
uv sync --extra gpu

# Add new packages
uv add package_name

# Add development packages  
uv add --dev package_name
```

### Testing
```bash
# Run tests with coverage
pytest --cov=whisperx_nemo_pipeline --cov-report=term-missing

# Run specific test markers
pytest -m "not slow"          # Skip slow tests
pytest -m gpu                 # GPU-only tests
pytest -m integration         # Integration tests

# Run tests in parallel
pytest -n auto

# Basic pipeline test
python test_pipeline.py

# Advanced pipeline test
python test_pipeline_adv.py
```

### Code Quality
```bash
# Format code
black whisperx_nemo_pipeline/ tests/

# Sort imports
isort whisperx_nemo_pipeline/ tests/

# Lint code
flake8 whisperx_nemo_pipeline/ tests/

# Type checking
mypy whisperx_nemo_pipeline/
```

### Build and Distribution
```bash
# Build package
python -m build

# Install in editable mode
uv pip install -e .
```

## Dependencies and External Packages

This project uses several external repositories and specialized packages:
- **ctc-forced-aligner**: From GitHub repo for word-level alignment
- **demucs**: Audio source separation (GitHub repo)
- **nemo_toolkit**: NVIDIA NeMo for speaker diarization
- **whisperx**: Enhanced Whisper implementation
- **faster-whisper**: Fast Whisper inference

## Configuration Files

- `pyproject.toml`: Main project configuration, dependencies, and tool settings
- `nemo_msdd_configs/diar_infer_telephonic.yaml`: NeMo diarization model configuration
- `uv.lock`: Lockfile for reproducible builds

## Key Design Patterns

- **Factory Pattern**: `create_transcription_pipeline()` for easy instantiation
- **Dataclass Configuration**: `TranscriptionConfig` for type-safe parameters
- **Parallel Processing**: Uses ProcessPoolExecutor for concurrent transcription/diarization
- **Resource Management**: Proper GPU memory cleanup and model lifecycle management
- **Fallback Handling**: Graceful degradation when optional dependencies are unavailable

## Testing Strategy

The project includes two main test files:
- `test_pipeline.py`: Basic pipeline functionality test
- `test_pipeline_adv.py`: Advanced scenarios and edge cases

Tests use real audio files and can be resource-intensive. Use pytest markers to control test execution based on available hardware and time constraints.