# run_whisper_asr.py
# downloads audio for a youtube video, trims it to a fixed duration,
# and transcribes it using openai whisper without hallucination loops.

import subprocess
import sys
import os
import torch
import whisper

script_dir = os.path.dirname(os.path.abspath(__file__))
asr_root = os.path.dirname(script_dir)
audio_dir = os.path.join(asr_root, "audio")
whisper_dir = os.path.join(asr_root, "whisper")

os.makedirs(audio_dir, exist_ok = True)
os.makedirs(whisper_dir, exist_ok = True)

def download_audio(youtube_video_url: str, output_file_path: str = None) -> str:

    if output_file_path is None:
        output_file_path = os.path.join(audio_dir, "audio.wav")

    # initiates the download process
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
        search_dir = os.path.dirname(output_file_path) or "."
        matching_files = [
            os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.startswith(os.path.basename(base_filename))
        ]
        if matching_files:
            output_file_path = matching_files[0]
        else:
            raise FileNotFoundError("could not locate downloaded audio file.")
            
    print(f"[ok] audio downloaded: {output_file_path}")
    return output_file_path


def trim_audio(input_file_path: str, duration_in_seconds: float, output_file_path: str = "trimmed.wav") -> str:
    # slices the audio file to match the evaluation window
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


def transcribe_audio(audio_file_path: str, whisper_model_size: str = "large-v3", target_language: str = "bn"):
    # identifies the hardware accelerator
    hardware_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] hardware device: {hardware_device}")
    
    if hardware_device == "cpu":
        print("[warning] running on cpu -- this will be computationally expensive.")

    print(f"[*] loading whisper model: {whisper_model_size} ...")
    try:
        whisper_model = whisper.load_model(whisper_model_size, device=hardware_device)
    except Exception as loading_error:
        print(f"[!] failed to load model: {loading_error}")
        raise

    print(f"[*] transcribing (language={target_language}) ...")
    try:
        # condition_on_previous_text=false forces the model to treat each segment independently,
        # stopping hallucination loops native to auto-regressive decoders.
        transcription_result = whisper_model.transcribe(
            audio_file_path,
            language=target_language,
            verbose=False,
            fp16=(hardware_device == "cuda"),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4
        )
    except torch.cuda.OutOfMemoryError as memory_error:
        print(f"[!] cuda out of memory: {memory_error}")
        print("    fallback to 'medium' model required for current vram allocation.")
        raise
    except Exception as execution_error:
        print(f"[!] transcription failed: {execution_error}")
        raise

    return transcription_result


def save_transcription_outputs(transcription_result, output_file_prefix: str):
    # writes the final strings to disk
    flat_text_path = f"{output_file_prefix}.txt"
    segments_text_path = f"{output_file_prefix}_segments.txt"

    with open(flat_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription_result["text"].strip())

    with open(segments_text_path, "w", encoding="utf-8") as segments_file:
        for segment in transcription_result["segments"]:
            segments_file.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'].strip()}\n")

    print(f"\nsaved flat asr transcript ->     {flat_text_path}")
    print(f"saved timestamped asr transcript -> {segments_text_path}")


def main():
    # executes the core pipeline
    if len(sys.argv) < 3:
        print("usage: python run_whisper_asr.py <youtube_url> <duration_seconds> [output_prefix] [model_size]")
        sys.exit(1)

    youtube_video_url = sys.argv[1]
    video_duration_seconds = float(sys.argv[2])
    output_file_prefix = sys.argv[3] if len(sys.argv) > 3 else "whisper_output"
    whisper_model_size = sys.argv[4] if len(sys.argv) > 4 else "medium"

    full_audio_file = download_audio(youtube_video_url)
    trimmed_audio_file = trim_audio(full_audio_file, video_duration_seconds, output_file_path=f"{output_file_prefix}_trimmed.wav")

    result = transcribe_audio(trimmed_audio_file, whisper_model_size=whisper_model_size, target_language="bn")
    save_transcription_outputs(result, output_file_prefix)


if __name__ == "__main__":
    main()