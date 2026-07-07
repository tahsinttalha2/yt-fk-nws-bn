"""
compute_wer.py

Computes Word Error Rate between your hand-corrected reference
transcript and an ASR hypothesis transcript, after applying
IDENTICAL normalization to both sides:

  - strip all punctuation (including Bangla dari "।")
  - unify numerals (Bangla digits <-> Arabic digits) to one form
  - collapse whitespace
  - (no lowercasing needed -- Bangla script has no case)

Normalizing both sides the same way avoids inflating WER with
differences that aren't real transcription errors (formatting,
punctuation, numeral style, stray speaker tags, etc).

Install:
    pip install jiwer

Usage:
    python compute_wer.py <reference.txt> <hypothesis.txt>

Example:
    python compute_wer.py youtube_transcript.txt asr_output.txt
"""

import re
import sys

from jiwer import wer, process_words


# Bangla digit -> Arabic digit mapping, used to unify numeral style
# before comparing (ASR output and manual captions don't always
# agree on which convention they use).
BANGLA_TO_ARABIC_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Characters to strip entirely: Bangla dari, standard punctuation,
# and any bracketed speaker/metadata tags like [Speaker 1].
PUNCTUATION_PATTERN = re.compile(r"[।,.!?;:\"'()\[\]{}—–\-]")
BRACKET_TAG_PATTERN = re.compile(r"\[[^\]]*\]")  # strips [Speaker 1], [MUSIC], etc.


def normalize(text: str) -> str:
    """Apply identical cleanup to reference and hypothesis before WER scoring."""
    text = BRACKET_TAG_PATTERN.sub(" ", text)
    text = text.translate(BANGLA_TO_ARABIC_DIGITS)
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    if len(sys.argv) != 3:
        print("Usage: python compute_wer.py <reference.txt> <hypothesis.txt>")
        sys.exit(1)

    ref_path, hyp_path = sys.argv[1], sys.argv[2]

    reference_raw = load_text(ref_path)
    hypothesis_raw = load_text(hyp_path)

    reference = normalize(reference_raw)
    hypothesis = normalize(hypothesis_raw)

    if not reference or not hypothesis:
        print("[!] One of the normalized transcripts is empty -- check your input files.")
        sys.exit(1)

    score = wer(reference, hypothesis)
    detail = process_words(reference, hypothesis)

    print("=" * 60)
    print(f"Reference file:  {ref_path}")
    print(f"Hypothesis file: {hyp_path}")
    print("=" * 60)
    print(f"WER: {score:.4f}  ({score * 100:.2f}%)")
    print("-" * 60)
    print(f"Substitutions: {detail.substitutions}")
    print(f"Deletions:     {detail.deletions}")
    print(f"Insertions:    {detail.insertions}")
    print(f"Hits:          {detail.hits}")
    print(f"Reference word count: {len(reference.split())}")
    print("=" * 60)
    print("\nNote: this WER reflects transcription accuracy only -- punctuation,")
    print("numeral style, and any speaker tags were stripped from BOTH sides")
    print("before comparison, so they do not count as errors either way.")


if __name__ == "__main__":
    main()
