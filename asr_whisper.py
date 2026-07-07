"""
Downloads audio for a YouTube video, trims it to a fixed duration
(matching your hand-corrected reference transcript window), and
transcribes it using Whisper.

Install:
    pip install yt-dlp openai-whisper
    (Whisper also needs ffmpeg installed and on PATH:
     https://www.gyan.dev/ffmpeg/builds/ -- grab a "full" build,
     unzip, add the bin folder to your PATH.)

Usage:
    python run_asr.py <youtube_url> <duration_seconds> [output_prefix]

Example (17:15.8 = 1035.8 seconds):
    python run_asr.py https://youtu.be/zu54yGYtZsA 1035.8 asr_output

Produces:
    asr_output.wav        -- trimmed audio (kept for re-runs / other models)
    asr_output.txt        -- flat ASR transcript (compare this against your reference)
    asr_output_segments.txt -- ASR transcript with per-segment timestamps
"""

import subprocess
import sys
import os


def download_audio(youtube_url: str, out_path: str = "audio.wav") -> str:
    """Download best audio track from YouTube as a WAV file via yt-dlp."""
    print(f"[*] Downloading audio from {youtube_url} ...")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", out_path.replace(".wav", ".%(ext)s"),
        youtube_url,
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(out_path):
        # yt-dlp sometimes names the output slightly differently; find it
        base = out_path.replace(".wav", "")
        candidates = [f for f in os.listdir(".") if f.startswith(os.path.basename(base))]
        if candidates:
            out_path = candidates[0]
        else:
            raise FileNotFoundError("Could not locate downloaded audio file.")
    print(f"[OK] Audio downloaded: {out_path}")
    return out_path


def trim_audio(in_path: str, duration_seconds: float, out_path: str = "trimmed.wav") -> str:
    """Trim audio to [0, duration_seconds] using ffmpeg."""
    print(f"[*] Trimming audio to first {duration_seconds} seconds ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-t", str(duration_seconds),
        "-ar", "16000",  # Whisper expects 16kHz
        "-ac", "1",      # mono
        out_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"[OK] Trimmed audio saved: {out_path}")
    return out_path


def transcribe(audio_path: str, model_size: str = "medium", language: str = "bn"):
    """Run Whisper on the trimmed audio. Returns the Whisper result dict."""
    import whisper  # openai-whisper

    print(f"[*] Loading Whisper model: {model_size} ...")
    model = whisper.load_model(model_size)

    print(f"[*] Transcribing (language={language}) ...")
    result = model.transcribe(audio_path, language=language, verbose=False)
    return result


def save_outputs(result, out_prefix: str):
    flat_path = f"{out_prefix}.txt"
    segments_path = f"{out_prefix}_segments.txt"

    with open(flat_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    with open(segments_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text'].strip()}\n")

    print(f"\nSaved flat ASR transcript ->     {flat_path}")
    print(f"Saved timestamped ASR transcript -> {segments_path}")
    print("\nUse the flat file as your WER hypothesis input (compute_wer.py).")


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_asr.py <youtube_url> <duration_seconds> [output_prefix]")
        sys.exit(1)

    youtube_url = sys.argv[1]
    duration_seconds = float(sys.argv[2])
    out_prefix = sys.argv[3] if len(sys.argv) > 3 else "asr_output"

    full_audio = download_audio(youtube_url)
    trimmed_audio = trim_audio(full_audio, duration_seconds, out_path=f"{out_prefix}_trimmed.wav")

    result = transcribe(trimmed_audio, model_size="medium", language="bn")
    save_outputs(result, out_prefix)


if __name__ == "__main__":
    main()
