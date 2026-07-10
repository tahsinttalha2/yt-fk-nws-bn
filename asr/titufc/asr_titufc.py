# asr_titufc.py
# downloads audio for a youtube video, trims it to a fixed duration,
# chunks the audio to prevent vram overflow, transcribes it using nemo,
# and applies text normalisation to reduce substitution errors.

import subprocess
import sys
import os
import re
import torch
import soundfile as sf

# --- path setup: location-independent, based on where this file physically sits ---
script_directory = os.path.dirname(os.path.abspath(__file__))
asr_root_directory = os.path.dirname(script_directory)
audio_directory = os.path.join(asr_root_directory, "audio")
titufc_directory = os.path.join(asr_root_directory, "titufc")

os.makedirs(audio_directory, exist_ok=True)
os.makedirs(titufc_directory, exist_ok=True)


def normalise_bangla_text(raw_text: str) -> str:
    # converts english numerals to bangla numerals and strips foreign characters
    english_to_bangla_digits = {
        '0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪',
        '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'
    }
    
    for english_digit, bangla_digit in english_to_bangla_digits.items():
        raw_text = raw_text.replace(english_digit, bangla_digit)
        
    normalised_text = re.sub(r'[a-zA-Z]', '', raw_text)
    normalised_text = re.sub(r'\s+', ' ', normalised_text).strip()
    
    return normalised_text


def download_youtube_audio(youtube_video_url: str, output_file_path: str = None) -> str:
    # initiates the download process for the requested url
    if output_file_path is None:
        output_file_path = os.path.join(audio_directory, "audio.wav")

    print(f"[*] downloading audio from {youtube_video_url} ...")
    command_arguments = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", output_file_path.replace(".wav", ".%(ext)s"),
        youtube_video_url,
    ]
    subprocess.run(command_arguments, check=True)

    if not os.path.exists(output_file_path):
        base_filename = output_file_path.replace(".wav", "")
        search_directory = os.path.dirname(output_file_path) or "."
        matching_files = [file for file in os.listdir(search_directory) if file.startswith(os.path.basename(base_filename))]
        if matching_files:
            output_file_path = os.path.join(search_directory, matching_files[0])
        else:
            raise FileNotFoundError("could not locate downloaded audio file.")
            
    print(f"[ok] audio downloaded: {output_file_path}")
    return output_file_path


def trim_downloaded_audio(input_file_path: str, duration_in_seconds: float, output_file_path: str = None) -> str:
    # slices the audio file to match the evaluation window
    if output_file_path is None:
        output_file_path = os.path.join(audio_directory, "trimmed.wav")
        
    print(f"[*] trimming audio to first {duration_in_seconds} seconds ...")
    command_arguments = [
        "ffmpeg", "-y",
        "-i", input_file_path,
        "-t", str(duration_in_seconds),
        "-ar", "16000",
        "-ac", "1",
        output_file_path,
    ]
    subprocess.run(command_arguments, check=True)
    print(f"[ok] trimmed audio saved: {output_file_path}")
    return output_file_path


def transcribe_audio_with_nemo(audio_file_path: str, nemo_model_identification: str = "hishab/titu_stt_bn_fastconformer"):
    import nemo.collections.asr as nemo_asr
    
    print(f"[*] loading nemo asr model: {nemo_model_identification} ...")
    try:
        speech_recogniser = nemo_asr.models.ASRModel.from_pretrained(model_name=nemo_model_identification)
    except Exception as loading_error:
        print(f"[!] failed to load nemo model: {loading_error}")
        raise
        
    print(f"[*] analysing audio duration for memory management ...")
    audio_data, sample_rate = sf.read(audio_file_path)
    total_samples = len(audio_data)
    
    # 20 seconds is a secure upper bound to prevent 6gb vram overflow
    chunk_duration_seconds = 20
    samples_per_chunk = chunk_duration_seconds * sample_rate
    
    temporary_chunk_files = []
    
    # slice the audio array into individual files
    for starting_index in range(0, total_samples, samples_per_chunk):
        chunk_array = audio_data[starting_index:starting_index + samples_per_chunk]
        chunk_filename = os.path.join(audio_directory, f"temp_chunk_{starting_index}.wav")
        sf.write(chunk_filename, chunk_array, sample_rate)
        temporary_chunk_files.append(chunk_filename)
        
    print(f"[*] split audio into {len(temporary_chunk_files)} smaller chunks to bypass vram limits ...")
    
    try:
        # process the chunks with a batch size of 2
        transcription_results = speech_recogniser.transcribe(temporary_chunk_files, batch_size=2)
    except Exception as execution_error:
        print(f"[!] transcription failed: {execution_error}")
        raise
    finally:
        # destroy temporary files to reclaim hard drive space
        for file_path in temporary_chunk_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                
    # extract strings, apply normalisation, and stitch together
    full_transcription = " ".join([normalise_bangla_text(result.text) for result in transcription_results])
    return full_transcription


def save_transcription_outputs(transcription_text: str, output_file_prefix: str):
    # writes the final string output to disk
    flat_text_path = os.path.join(titufc_directory, f"{output_file_prefix}.txt")

    with open(flat_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription_text.strip())

    print(f"\nsaved flat asr transcript -> {flat_text_path}")


def main():
    # executes the core pipeline
    if len(sys.argv) < 3:
        print("usage: python asr_titufc.py <youtube_url> <duration_seconds> [output_prefix] [model_id]")
        sys.exit(1)

    youtube_video_url = sys.argv[1]
    video_duration_seconds = float(sys.argv[2])

    output_file_prefix = sys.argv[3] if len(sys.argv) > 3 else "titufc_output"
    huggingface_model_identification = sys.argv[4] if len(sys.argv) > 4 else "hishab/titu_stt_bn_fastconformer"

    if huggingface_model_identification in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "sazzadul/Shrutimala_Bangla_ASR"]:
        print(f"[*] notice: '{huggingface_model_identification}' is incorrect for this script. defaulting to fastconformer.")
        huggingface_model_identification = "hishab/titu_stt_bn_fastconformer"

    full_audio_file = download_youtube_audio(youtube_video_url)

    trimmed_audio_file = trim_downloaded_audio(
        full_audio_file,
        video_duration_seconds,
        output_file_path=os.path.join(audio_directory, f"{output_file_prefix}_trimmed.wav"),
    )

    final_result = transcribe_audio_with_nemo(
        trimmed_audio_file,
        nemo_model_identification=huggingface_model_identification
    )
    
    save_transcription_outputs(final_result, output_file_prefix)


if __name__ == "__main__":
    main()