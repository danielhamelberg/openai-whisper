import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import logging
from datetime import datetime

# Set up logging with a log file in the current directory
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def seconds_to_srt_time_format(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def save_transcription_to_srt(segments, audio_file):
    try:
        audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
        current_date = datetime.now().strftime("%Y%m%d")
        srt_filename = f"{current_date}_{audio_file_name}.srt"
        logger.info(f"Saving transcription to: {srt_filename}")
        with open(srt_filename, 'w') as srt_file:
            for index, segment in enumerate(segments):
                start_time = seconds_to_srt_time_format(segment['timestamp'][0])
                end_time = seconds_to_srt_time_format(segment['timestamp'][1])
                srt_file.write(f"{index + 1}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{segment['text'].strip()}\n\n")
                logger.debug(f"Segment {index + 1}: {start_time} --> {end_time} | Text: {segment['text'].strip()}")
        logger.info(f"Transcription saved to {srt_filename}")
    except IOError as e:
        logger.error(f"An I/O error occurred while saving the transcription: {e}")
    except ValueError as e:
        logger.error(f"A value error occurred while saving the transcription: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the transcription: {e}")

def main():
    # List all files in the 'audio_sources' directory
    try:
        directory_contents = os.listdir("audio_sources")
    except FileNotFoundError:
        logger.error("Directory 'audio_sources' not found.")
        return

    logger.debug("Gathering audio files from 'audio_sources'...")
    audio_files = [
        os.path.join("audio_sources", f)
        for f in directory_contents
        if f.endswith(".mp3")
    ]

    if not audio_files:
        logger.warning("No audio files found in 'audio_sources'.")
        return

    logger.info(f"Found {len(audio_files)} audio file(s) to process.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = os.getenv("MODEL_ID")
    if model_id is None:
        logger.error("MODEL_ID environment variable not set")
        raise ValueError("MODEL_ID environment variable not set")
    logger.info(f"Using model ID: {model_id}")

    logger.debug("Initializing model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    logger.debug("Model loaded and moved to device.")

    logger.debug("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    logger.debug("Processor loaded.")

    logger.debug("Building the pipeline...")
    language_code = os.getenv("LANGUAGE")
    if language_code is None:
        logger.error("LANGUAGE_CODE environment variable not set")
        raise ValueError("LANGUAGE_CODE environment variable not set")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=4,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        generate_kwargs={"language": language_code}
    )
    logger.debug("Pipeline created.")

    for i, audio_file in enumerate(audio_files, start=1):
        logger.info(f"Processing file {i} of {len(audio_files)}: {audio_file}")
        try:
            result = pipe(audio_file, return_timestamps=True, generate_kwargs={"language": "nl"})
            logger.debug(f"Raw pipeline result: {result}")

            if "chunks" in result:
                transcription = result["chunks"]
                logger.debug("Chunks detected in the pipeline result.")
            else:
                logger.debug("No chunks detected, transforming pipeline result manually.")
                transcription = [{
                    "timestamp": seg["timestamp"], 
                    "text": seg.get("text", "")
                } for seg in result]

            save_transcription_to_srt(transcription, audio_file)

            # Move the file after successful transcription
            processed_path = os.path.join("audio_sources_processed", os.path.basename(audio_file))
            os.rename(audio_file, processed_path)
            logger.info(f"Moved processed file to: {processed_path}")
        except Exception as exc:
            logger.error(f"Error processing file {audio_file}: {exc}")

if __name__ == "__main__":
    main()
