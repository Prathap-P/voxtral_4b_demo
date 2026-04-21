"""
Voxtral-4B-TTS v5.1 — Volume Decay Fix
Fixes v5's volume fade-out within long chunks (attention decay over 60-90s):
  - Reduced chunk size to 1500 chars (~30-45s audio, within attention sweet spot)
  - Per-chunk RMS normalization to level out volume before stitching
  - Lighter noise reduction (0.3) to avoid stripping quiet end-of-chunk audio
  - Still: no overlap, no silence trimming, silence-gap stitching, paragraph gaps

Model: mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 (~8GB, best quality)
Output: output1.wav (24kHz, lossless)
"""

import gc
import re
import sys
import time
import numpy as np
import soundfile as sf
import noisereduce as nr
from mlx_audio.tts.utils import load

# ─── Configuration ───────────────────────────────────────────────────────────
# bf16 = full precision, best quality, no quantization artifacts (~8GB)
MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"

VOICE = "neutral_male"
OUTPUT_FILE = "output1.wav"
SAMPLE_RATE = 24000              # Voxtral native sample rate
MAX_CHUNK_CHARS = 1500           # Balance: few boundaries (8-10) but short enough to avoid volume decay

# Silence gaps for stitching chunks together
SENTENCE_GAP_MS = 300            # Silence gap between regular chunks
PARAGRAPH_GAP_MS = 700           # Longer silence gap at paragraph boundaries

# Generation parameters — lower temp = more consistent voice across chunks
TEMPERATURE = 0.4                # Default is 0.8; lower = less stochastic voice drift
TOP_K = 30                       # Default is 50; tighter sampling for consistency
TOP_P = 0.90                     # Default is 0.95; slightly tighter nucleus sampling

# Post-processing
ENABLE_NOISE_REDUCTION = True    # Set False to compare raw vs cleaned output
NOISE_REDUCE_STRENGTH = 0.3     # 0.0 = no reduction, 1.0 = aggressive (0.3 preserves quiet passages)

# ─── Text ────────────────────────────────────────────────────────────────────
# Replace this with your own text (supports 20K-30K+ characters)
TEXT = (
    """Concerns are rising about other nations using AI to remove humans from decision-making. The Pentagon moved to ban certain providers like Anthropic earlier this year... due to supply chain risks. This prompted a discussion with Under Secretary Emil Michael about how AI might change warfare... He draws an analogy to the ride-sharing industry. Safety statistics for systems like Tesla FSD are actually amazing. The fear is of change itself... in reality, the technology makes service more reliable and precise. Much like how Uber reduced drinking driving while increasing availability... applying this to military contexts means being able to discern a decoy from a non-decoy within drone swarms. A recent demonstration by Cameron Stanley... showcased this through a program called Maven Smart System. Specifically, the Target Workbench allows users to view live images and select targets within a unified workflow. It is not about replacing human judgment... but increasing the "human context window" by synthesizing vast amounts of data.

When a target is identified, the system calculates variables such as weather conditions... fuel consumption, and collateral effects. It does not operate like a chatbot or Skynet; instead, it serves as an orchestration layer on top of data streams. The Under Secretary clarifies that no Large Language Model is baked into the kill chain itself... countering common misconceptions about automated killing. Instead, tools like Palantir surface choices that are otherwise consumed by spreadsheets and PowerPoint files—methods historically used to relay target lists. The digitalization of targeting processes accelerates these decisions, granting a single operator the power of many more. While permissions and authorities remain strictly human-controlled to ensure checks and balances, the system provides better outcomes through informed clicks. This shift from manual coordination to AI-assisted synthesis represents a responsible evolution of war fighting... moving beyond the chaos of unconnected data to a unified strategy.

The discussion outlines three layers of artificial intelligence application within defense, starting with efficiency... Mundane work is streamlined so personnel can focus on more interesting tasks. Then there is the intelligence layer... Imagine all the intelligence gathered from satellite imagery worldwide. Currently, a human analyst must look at everything to make a judgment... but with historical data and AI synthesis, the system can identify anomalies. It learns what those anomalies are—creating a totally different paradigm for intelligence analysis if you will. Moving on to the third layer, war fighting... AI takes all paperwork and modeling and simulation to react faster. But also more precisely.
Moving to conflicts of interest, the host queried about XAI and SpaceX holdings. The official confirmed he sold all SpaceX stock to comply with the Office of Government Ethics... Defense company stocks are red lines, and he recused himself from XAI dealings until the sale cleared. Next came procurement reform involving Uber and Palantir dynamics... Defense contractors consolidated from fifty down to five since the Cold War, making supply chains brittle.

The official argued shifting from cost-plus contracts to performance-based deals is necessary... If a weapon works on time, they get paid; if not, they do not. This risk-sharing model benefits taxpayers and encourages innovation without massive speculative R&D burdens... Founders like Palmer Lucky are willing to enter this business finally. Finally, the conversation addressed the Pentagon Pizza Index tracking orders to predict military action... The official dismissed this entirely stating he has no idea how food gets delivered inside the Pentagon building. There are no specific Papa John's locations delivering directly in... The segment concluded by thanking guests for visiting Washington DC. This occurred before signing off on Big Technology Podcast.""")


# ─── Sentence Splitter ──────────────────────────────────────────────────────
def split_into_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences at . ! ? ... boundaries.
    Handles common abbreviations and ellipsis patterns.
    """
    # Split at sentence-ending punctuation followed by whitespace
    raw_parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    # Filter out empty strings
    return [s.strip() for s in raw_parts if s.strip()]


# ─── Chunking ──────────────────────────────────────────────────────────────
def build_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """
    Build chunks at sentence boundaries with paragraph break tracking.

    Each chunk includes:
      - 'text': The text to generate audio for
      - 'has_paragraph_break': Whether a paragraph break occurs before the NEXT chunk

    Simple chunking — no overlap conditioning. With 1500-char chunks there are
    ~8-10 boundaries — balanced to avoid both voice drift and volume decay.
    """
    # Split into paragraphs, then sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    all_sentences = []
    paragraph_break_indices = set()  # Track where paragraph breaks occur for natural pauses

    for para in paragraphs:
        if all_sentences:
            paragraph_break_indices.add(len(all_sentences))
        sentences = split_into_sentences(para)
        all_sentences.extend(sentences)

    if not all_sentences:
        return [{"text": text, "has_paragraph_break": False}]

    chunks = []
    current_sentences = []
    current_len = 0

    for i, sentence in enumerate(all_sentences):
        # Would adding this sentence exceed the limit?
        added_len = len(sentence) + (1 if current_len > 0 else 0)

        if current_len + added_len > max_chars and current_sentences:
            # Flush current chunk
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text,
                "has_paragraph_break": i in paragraph_break_indices,
            })

            current_sentences = [sentence]
            current_len = len(sentence)
        else:
            current_sentences.append(sentence)
            current_len += added_len

    # Flush remaining sentences
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "text": chunk_text,
            "has_paragraph_break": False,
        })

    return chunks


# ─── Audio Processing Utilities ──────────────────────────────────────────────
def create_silence(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Create a silence array of the specified duration."""
    return np.zeros(int(sample_rate * duration_ms / 1000), dtype=np.float32)


def rms_normalize(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """
    Normalize audio to a target RMS level (perceived loudness matching).
    RMS normalization is better than peak normalization for consistent
    perceived volume across chunks.
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        gain = target_rms / current_rms
        # Clip to prevent extreme gains on very quiet chunks
        gain = min(gain, 10.0)
        normalized = audio * gain
        # Safety clip to prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)
        return normalized
    return audio


def reduce_noise(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                 strength: float = NOISE_REDUCE_STRENGTH) -> np.ndarray:
    """
    Apply spectral gating noise reduction to remove background noise (wind, hiss).
    """
    reduced = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=True,
        prop_decrease=strength,
        n_fft=1024,
        freq_mask_smooth_hz=200,
        time_mask_smooth_ms=100,
    )
    return reduced.astype(np.float32)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


# ─── Main Generation ─────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Voxtral-4B-TTS v5.1 — Volume Decay Fix")
    print("  Fixes: volume fade-out, natural pauses, voice consistency")
    print("=" * 70)

    # Build chunks at sentence boundaries
    chunks = build_chunks(TEXT)
    total_chars = sum(len(c["text"]) for c in chunks)

    print(f"\n  Model:          {MODEL_ID}")
    print(f"  Voice:          {VOICE}")
    print(f"  Temperature:    {TEMPERATURE} (lower = more consistent voice)")
    print(f"  Top-k / Top-p:  {TOP_K} / {TOP_P}")
    print(f"  Text:           {len(TEXT):,} characters")
    print(f"  Chunks:         {len(chunks)} (max {MAX_CHUNK_CHARS} chars, sentence boundaries)")
    print(f"  Stitching:      Silence gaps ({SENTENCE_GAP_MS}ms sentence / {PARAGRAPH_GAP_MS}ms paragraph)")
    print(f"  Noise reduce:   {'ON (strength={})'.format(NOISE_REDUCE_STRENGTH) if ENABLE_NOISE_REDUCTION else 'OFF'}")
    print(f"  Normalization:  Per-chunk RMS + final global RMS (fixes volume decay)")
    print(f"  Output:         {OUTPUT_FILE}")
    print()

    # Load model
    print("Loading model (bf16 = ~8GB, first run downloads from HuggingFace)...")
    start_load = time.time()
    model = load(MODEL_ID)
    load_time = time.time() - start_load
    print(f"Model loaded in {format_time(load_time)}")

    # Generate speech chunk by chunk
    print(f"\nGenerating speech ({len(chunks)} chunks, no overlap conditioning)...")
    print("-" * 70)

    start_gen = time.time()
    all_audio_segments = []
    chunk_metadata = []  # Track has_paragraph_break for assembly
    chars_done = 0

    for i, chunk_info in enumerate(chunks):
        chunk_start = time.time()
        chunk_text = chunk_info["text"]
        has_para_break = chunk_info["has_paragraph_break"]

        chars_done += len(chunk_text)
        progress_pct = (chars_done / total_chars) * 100

        # Show progress
        preview = chunk_text[:55].replace("\n", " ")
        if len(chunk_text) > 55:
            preview += "..."
        print(f"\n  [{i+1}/{len(chunks)}] ({progress_pct:5.1f}%) \"{preview}\"")
        sys.stdout.flush()

        # Generate audio for chunk with controlled sampling
        chunk_audio_parts = []
        for result in model.generate(
            text=chunk_text,
            voice=VOICE,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
        ):
            chunk_audio_parts.append(np.array(result.audio))

        if not chunk_audio_parts:
            print(f"           WARNING: No audio generated for chunk {i+1}")
            continue

        chunk_audio = np.concatenate(chunk_audio_parts) if len(chunk_audio_parts) > 1 else chunk_audio_parts[0]

        if len(chunk_audio) == 0:
            print(f"           WARNING: Chunk {i+1} produced empty audio")
            continue

        # Per-chunk RMS normalization — fixes volume decay within long chunks
        # (attention to voice embedding fades as sequence grows, causing quieter end-of-chunk audio)
        chunk_audio = rms_normalize(chunk_audio)

        chunk_duration = len(chunk_audio) / SAMPLE_RATE
        chunk_time = time.time() - chunk_start
        elapsed = time.time() - start_gen
        eta = (elapsed / chars_done) * (total_chars - chars_done) if chars_done > 0 else 0

        print(f"           Done: {format_time(chunk_duration)} audio | "
              f"took {format_time(chunk_time)} | "
              f"ETA: {format_time(eta)}")

        all_audio_segments.append(chunk_audio)
        chunk_metadata.append({"has_paragraph_break": has_para_break})

        # Periodic memory cleanup
        if (i + 1) % 10 == 0:
            gc.collect()

    gen_time = time.time() - start_gen

    if not all_audio_segments:
        print("\nERROR: No audio segments generated!")
        return

    # Assemble final audio with silence gaps
    print(f"\nAssembling {len(all_audio_segments)} segments with silence-gap stitching...")
    if len(all_audio_segments) == 1:
        audio_full = all_audio_segments[0]
    else:
        parts = [all_audio_segments[0]]
        for idx in range(1, len(all_audio_segments)):
            # Check if this chunk starts a new paragraph (previous chunk flagged the break)
            if chunk_metadata[idx - 1]["has_paragraph_break"]:
                gap = create_silence(PARAGRAPH_GAP_MS)
            else:
                gap = create_silence(SENTENCE_GAP_MS)
            parts.append(gap)
            parts.append(all_audio_segments[idx])
        audio_full = np.concatenate(parts)

    # Post-processing: Noise Reduction
    if ENABLE_NOISE_REDUCTION:
        print(f"Applying noise reduction (strength={NOISE_REDUCE_STRENGTH})...")
        nr_start = time.time()
        audio_full = reduce_noise(audio_full)
        nr_time = time.time() - nr_start
        print(f"  Noise reduction done in {format_time(nr_time)}")

    # Final global RMS normalization of assembled audio
    audio_full = rms_normalize(audio_full)

    # Safety: clip to prevent any floating-point overflow
    audio_full = np.clip(audio_full, -1.0, 1.0)

    # Save as WAV
    sf.write(OUTPUT_FILE, audio_full, SAMPLE_RATE)

    # ─── Results ─────────────────────────────────────────────────────────
    duration = len(audio_full) / SAMPLE_RATE
    file_size_mb = (len(audio_full) * 4) / (1024 * 1024)

    print(f"\n{'=' * 70}")
    print(f"  Generation Complete! (v5.1 — Volume Decay Fix)")
    print(f"{'=' * 70}")
    print(f"  File:            {OUTPUT_FILE}")
    print(f"  File size:       {file_size_mb:.1f} MB")
    print(f"  Audio duration:  {format_time(duration)}")
    print(f"  Generation time: {format_time(gen_time)}")
    print(f"  Real-time factor: {gen_time/duration:.2f}x (lower is faster)")
    print(f"  Sample rate:     {SAMPLE_RATE} Hz")
    print(f"  Chunks used:     {len(chunks)}")
    print(f"  Characters:      {total_chars:,}")
    print(f"  Temperature:     {TEMPERATURE}")
    print(f"  Noise reduction: {'ON' if ENABLE_NOISE_REDUCTION else 'OFF'}")
    print(f"  Stitching:       Silence gaps ({SENTENCE_GAP_MS}ms / {PARAGRAPH_GAP_MS}ms)")
    print(f"{'=' * 70}")
    print(f"\nOpen '{OUTPUT_FILE}' in any audio player to listen!")
    print(f"Compare with previous version to hear the improvement.")


if __name__ == "__main__":
    main()
