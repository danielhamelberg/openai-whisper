# Audio Transcription Pipeline

This project processes audio files from the `audio_sources` directory, transcribes them using a specified transcription pipeline, and saves the transcriptions in SRT format. Processed audio files are moved to the `audio_sources_processed` directory.

## Project Structure

- `audio_sources/`: Directory containing the input audio files.
- `audio_sources_processed/`: Directory where processed audio files are moved.
- `docker-compose.yml`: Docker Compose configuration for running the transcription pipeline.
- `docker-compose-ffmpeg.yml`: Docker Compose configuration for converting audio files to MP3 format.
- `Dockerfile.transformers`: Dockerfile for setting up the transcription environment.
- `infer.py`: Main script for processing and transcribing audio files.
- `requirements.txt`: Python dependencies.

## Setup

1. **Install Dependencies**: Ensure you have Python installed, then install the required packages using pip:
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the Transcription Pipeline**: Execute the [infer.py](https://github.com/danielhamelberg/openai-whisper/infer.py) script to process and transcribe audio files:
    ```sh
    python infer.py
    ```

## Using Docker

1. **Build and Run with Docker Compose**:
    ```sh
    docker-compose up --build
    ```

2. **Convert Audio Files to MP3**:
    ```sh
    docker-compose -f docker-compose-ffmpeg.yml up
    ```

## Environment Variables

- `MODEL_ID`: Specify the model ID for the transcription pipeline (e.g., `openai/whisper-large-v3-turbo`).
- `LANGUAGE`: Specify the language code of the input audio (e.g., `en`, or `nl`).

## Logging

Logs are saved to [transcription.log] in the project directory.
