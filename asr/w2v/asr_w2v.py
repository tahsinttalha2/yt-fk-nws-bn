# asr_titufc.py
# downloads audio, maps word boundaries using whisper to avoid mid-word slicing,
# aggressively manages vram, transcribes via nemo, and normalises text.

import subprocess
import sys
import os
import re
import gc
import torch
import soundfile as sf
import whisper

# --- path setup: location-independent ---
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


def map_word_boundaries_and_chunk(audio_file_path: str) -> list:
    # uses a lightweight whisper model to find word boundaries and slice audio sequentially
    print("[*] mapping word boundaries using lightweight whisper base to avoid mid-word slices ...")
    
    # load a small model to preserve vram
    alignment_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    alignment_result = alignment_model.transcribe(audio_file_path, language="bn", word_timestamps=True)
    
    word_end_times = []
    for segment in alignment_result.get("segments", []):
        for word in segment.get("words", []):
            word_end_times.append(word["end"])
            
    # aggressively clear vram before loading the conformer
    del alignment_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[ok] word boundaries mapped. whisper purged from vram.")

    print("[*] slicing audio at semantic boundaries ...")
    audio_data, sample_rate = sf.read(audio_file_path)
    total_samples = len(audio_data)
    
    semantic_chunk_files = []
    current_start_sample = 0
    target_chunk_seconds = 20.0
    current_target = target_chunk_seconds
    
    for end_time in word_end_times:
        if end_time >= current_target:
            end_sample = int(end_time * sample_rate)
            end_sample = min(end_sample, total_samples)
            
            chunk_array = audio_data[current_start_sample:end_sample]
            chunk_duration = (end_sample - current_start_sample) / sample_rate
            
            # discards chunks shorter than 2 seconds as they lack acoustic context
            if chunk_duration >= 2.0:
                chunk_filename = os.path.join(audio_directory, f"semantic_chunk_{current_start_sample}.wav")
                sf.write(chunk_filename, chunk_array, sample_rate)
                semantic_chunk_files.append(chunk_filename)
                
            current_start_sample = end_sample
            current_target = end_time + target_chunk_seconds
            
    # handles the remaining audio tail
    if current_start_sample < total_samples:
        chunk_array = audio_data[current_start_sample:]
        if len(chunk_array) / sample_rate >= 2.0:
            chunk_filename = os.path.join(audio_directory, f"semantic_chunk_{current_start_sample}.wav")
            sf.write(chunk_filename, chunk_array, sample_rate)
            semantic_chunk_files.append(chunk_filename)
            
    print(f"[ok] generated {len(semantic_chunk_files)} context-aware chunks.")
    return semantic_chunk_files


def transcribe_chunks_with_nemo(chunk_files: list, nemo_model_identification: str = "hishab/titu_stt_bn_fastconformer"):
    import nemo.collections.asr as nemo_asr
    
    print(f"[*] loading nemo asr model: {nemo_model_identification} ...")
    try:
        speech_recogniser = nemo_asr.models.ASRModel.from_pretrained(model_name=nemo_model_identification)
    except Exception as loading_error:
        print(f"[!] failed to load nemo model: {loading_error}")
        raise
        
    print(f"[*] transcribing chunks with safe batch sizing ...")
    try:
        # process chunks with batch size 2 to prevent vram overflow
        transcription_results = speech_recogniser.transcribe(chunk_files, batch_size=2)
    except Exception as execution_error:
        print(f"[!] transcription failed: {execution_error}")
        raise
    finally:
        # destroy temporary files to reclaim hard drive space
        for file_path in chunk_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                
    # normalise and stitch
    full_transcription = " ".join([normalise_bangla_text(result.text) for result in transcription_results])
    return full_transcription


def save_transcription_outputs(transcription_text: str, output_file_prefix: str):
    # writes the final string output to disk
    flat_text_path = os.path.join(titufc_directory, f"{output_file_prefix}.txt")

    with open(flat_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription_text.strip())

    print(f"\nsaved flat asr transcript -> {flat_text_path}")


def main():
    # executes the sequential pipeline
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

    # step 1: word-boundary alignment via lightweight whisper
    semantic_chunks = map_word_boundaries_and_chunk(trimmed_audio_file)
    
    # step 2: heavy transcription via nemo
    final_result = transcribe_chunks_with_nemo(
        semantic_chunks,
        nemo_model_identification=huggingface_model_identification
    )
    
    save_transcription_outputs(final_result, output_file_prefix)


if __name__ == "__main__":
    main()