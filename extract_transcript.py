"""
extract_transcript.py

Fetches a YouTube video's caption track and reports whether it is
auto-generated (ASR) or manually created, BEFORE you use it as a
WER gold-reference. Auto-generated captions carry the same error
profile as any other ASR output and are not safe as ground truth.

Written for youtube-transcript-api v1.x (instance-based API).

Install:
    pip install youtube-transcript-api

Usage:
    python extract_transcript.py <youtube_url_or_id> [output_file.txt]

Example:
    python extract_transcript.py https://youtu.be/zu54yGYtZsA transcript.txt
"""

import re
import sys

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


def extract_video_id(url_or_id: str) -> str:
    """Pull the 11-character YouTube video ID out of a URL, or return it as-is."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # watch?v=... or /embed/...
        r"youtu\.be\/([0-9A-Za-z_-]{11})",  # youtu.be short links
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"Could not extract a video ID from: {url_or_id}")


def list_available_tracks(ytt_api: YouTubeTranscriptApi, video_id: str):
    """Print every available caption track and whether each is ASR or manual."""
    try:
        transcript_list = ytt_api.list(video_id)
    except TranscriptsDisabled:
        print(f"[!] Captions are disabled for this video: {video_id}")
        return None
    except Exception as e:
        print(f"[!] Could not fetch transcript list: {e}")
        return None

    print(f"\nAvailable caption tracks for video: {video_id}")
    print("-" * 60)
    tracks = list(transcript_list)
    for t in tracks:
        kind = "AUTO-GENERATED (ASR)" if t.is_generated else "MANUAL"
        print(f"  {t.language} ({t.language_code}) | {kind} | translatable: {t.is_translatable}")

    if not tracks:
        print("  (none found)")
    return transcript_list


def fetch_best_transcript(ytt_api: YouTubeTranscriptApi, video_id: str, preferred_langs=("bn", "en")):
    """
    Fetch a transcript, preferring MANUAL tracks over ASR tracks,
    and preferring Bangla over English when both exist.
    Returns (chosen_track, fetched_transcript) or None.
    """
    try:
        transcript_list = ytt_api.list(video_id)
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"[!] No transcript available: {e}")
        return None

    all_tracks = list(transcript_list)
    manual_tracks = [t for t in all_tracks if not t.is_generated]
    auto_tracks = [t for t in all_tracks if t.is_generated]

    def pick_by_lang(tracks):
        for lang in preferred_langs:
            for t in tracks:
                if t.language_code.startswith(lang):
                    return t
        return tracks[0] if tracks else None

    chosen = pick_by_lang(manual_tracks) or pick_by_lang(auto_tracks)

    if chosen is None:
        print("[!] No usable transcript found.")
        return None

    kind = "ASR (auto-generated)" if chosen.is_generated else "MANUAL"
    print(f"\n[Selected track] {chosen.language} ({chosen.language_code}) | {kind}")

    if chosen.is_generated:
        print("[WARNING] This is an auto-generated caption track.")
        print("          Do NOT use it as a WER gold-reference -- it has")
        print("          the same error profile as the ASR you're testing.")
    else:
        print("[OK] Manually created caption track -- safe to consider as")
        print("     a WER reference, pending your own by-ear spot check")
        print("     for paraphrasing / translation of code-switched words.")

    fetched = chosen.fetch()  # FetchedTranscript object (iterable of snippets)
    return chosen, fetched


def save_transcript(fetched, out_path="transcript.txt"):
    """Save a plain-text version (for WER comparison) and a timestamped version (for review)."""
    timed_path = out_path.rsplit(".", 1)[0] + "_timed.txt"

    with open(out_path, "w", encoding="utf-8") as f_plain, \
         open(timed_path, "w", encoding="utf-8") as f_timed:
        for snippet in fetched:  # snippet is a FetchedTranscriptSnippet: .text, .start, .duration
            text = snippet.text.replace("\n", " ").strip()
            if not text:
                continue
            f_plain.write(text + " ")
            f_timed.write(f"[{snippet.start:.2f}s] {text}\n")

    print(f"\nSaved plain transcript ->      {out_path}")
    print(f"Saved timestamped transcript -> {timed_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_transcript.py <youtube_url_or_id> [output_file.txt]")
        sys.exit(1)

    url_or_id = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "transcript.txt"

    video_id = extract_video_id(url_or_id)
    ytt_api = YouTubeTranscriptApi()

    list_available_tracks(ytt_api, video_id)

    result = fetch_best_transcript(ytt_api, video_id)
    if result is None:
        sys.exit(1)

    _chosen, fetched = result
    save_transcript(fetched, out_path)


if __name__ == "__main__":
    main()