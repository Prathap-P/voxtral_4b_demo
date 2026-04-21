# Voxtral-4B-TTS Knowledge Base

## Model Identity
- **Model (max quality)**: `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` (~8GB, best quality, no quantization artifacts)
- **Model (recommended fast)**: `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` (~3.5GB, ~1.33x real-time on M4 Pro, near-bf16 quality, no noise artifacts)
- **Lightweight alternative**: `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` (~2.5GB, fastest but produces wind/noise artifacts — see "Quantization Artifacts" section)
- **Original weights**: `mistralai/Voxtral-4B-TTS-2603` (requires vLLM-Omni, not usable with mlx-audio)
- **Architecture**: 4.1B parameters — 3-stage pipeline: transformer decoder backbone (3.4B) → flow-matching acoustic transformer (390M) → neural audio codec decoder (300M)
- **Quantization**: 4-bit (~2.5GB), 6-bit (~3.5GB, best speed/quality trade-off), bf16 (~8GB, max quality)
- **License**: CC BY-NC 4.0 (non-commercial only, commercial via Mistral API)
- **Output**: 24kHz audio, supports WAV/PCM/FLAC/MP3/AAC/Opus

## Python Library
- **Package**: `mlx-audio` (version 0.4.2+)
- **Extra dependencies**: `tiktoken` (Tekken tokenizer), `noisereduce` (post-processing noise cleanup)
- **Install**: `pip install mlx-audio[tts] soundfile numpy tiktoken noisereduce`
- **Platform**: Apple Silicon only (M1/M2/M3/M4), uses Metal GPU via MLX framework
- **Import**: `from mlx_audio.tts.utils import load`

## Selected Voices
Three voices selected for production use:

| Voice ID | Gender | Tone | Best For |
|----------|--------|------|----------|
| `casual_female` | Female | Relaxed, conversational | Informal narration, friendly dialogue |
| `neutral_male` | Male | Balanced, professional | News reading, informational content, reports |
| `neutral_female` | Female | Balanced, professional | Announcements, instructional content |

### All 20 Available Voice Presets (for reference)
**English**: `casual_male`, `casual_female`, `cheerful_female`, `neutral_male`, `neutral_female`
**French**: `fr_male`, `fr_female`
**Spanish**: `es_male`, `es_female`
**German**: `de_male`, `de_female`
**Italian**: `it_male`, `it_female`
**Portuguese**: `pt_male`, `pt_female`
**Dutch**: `nl_male`, `nl_female`
**Arabic**: `ar_male`
**Hindi**: `hi_male`, `hi_female`

## Core Generation API

### Load Model (once, reuse across generations)
```python
from mlx_audio.tts.utils import load

# Choose one — 6-bit recommended for speed + quality balance
model = load("mlx-community/Voxtral-4B-TTS-2603-mlx-6bit")    # ~3.5GB, ~1.33x real-time, clean audio
# model = load("mlx-community/Voxtral-4B-TTS-2603-mlx-bf16")  # ~8GB, slowest, max quality
```
- Model auto-downloads from HuggingFace on first call (~3.5GB for 6-bit, ~8GB for bf16, ~2.5GB for 4-bit)
- HuggingFace Hub handles caching — subsequent loads are instant (~2-3s)
- Keep model instance alive and reuse it — do not reload per request
- 6-bit model only quantizes the LLM backbone — acoustic transformer and codec decoder stay at full precision, so no noise artifacts

### Generate Audio
```python
import numpy as np

audio_parts = []
for result in model.generate(text="Your text here", voice="casual_female"):
    audio_parts.append(np.array(result.audio))

audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
```
- `model.generate()` is a **generator** — yields one or more `result` objects
- Each `result.audio` is an `mx.array` (MLX tensor) — convert to numpy with `np.array()`
- Multiple results may be yielded for longer text — always collect and concatenate
- `result.audio` contains float32 samples at 24kHz

### Save to WAV
```python
import soundfile as sf

sf.write("output.wav", audio, 24000)
```
- Sample rate is always 24000 Hz
- WAV format works without ffmpeg
- Audio is float32, soundfile handles the encoding

## Long Text Handling (Critical for 20K-30K+ characters)

### Why Chunking is Required
- `max_tokens=4096` (default) supports ~5.5 minutes of audio (4096 × 80ms per frame)
- Model was trained on audio up to **180 seconds** — longer sequences degrade
- Each `model.generate()` call creates a **fresh KV cache** — no context between calls
- Memory usage spikes on long sequences — can OOM on constrained devices

### Root Cause of Chunk Boundary Issues
Each independent `model.generate(text=chunk, voice=VOICE)` call:
1. Creates a **brand new KV cache** (line 607 in voxtral_tts.py)
2. Encodes text from scratch with no knowledge of what came before
3. Uses random sampling (default temp=0.8) causing stochastic voice drift

**This causes**: voice drift within chunks and inconsistent voice across chunks. Minimizing the number of chunk boundaries is the most effective mitigation.

### Solution: Balanced Chunks (v5.1)
Use **1500-character chunks** at sentence boundaries. This balances two competing concerns:
- **Too many boundaries** (300 chars / v4) → voice drift at every boundary
- **Too few boundaries** (2500 chars / v5) → volume decays within each chunk as attention dilutes

```
Text (12,500 chars) → 8-10 chunks of ~1500 chars each → 7-9 boundaries (v5.1 sweet spot)
Text (12,500 chars) → 5 chunks of ~2500 chars each → 4 boundaries (v5 — volume decay)
Text (12,500 chars) → 42 chunks of ~300 chars each → 41 boundaries (v4 — voice drift)
```

### Chunking Strategy (v5)
Split text at **sentence boundaries only** (never mid-sentence), tracking paragraph breaks:
1. Split text into paragraphs, then split each paragraph into sentences
2. Group sentences into chunks of **1500 characters max**
3. Track `has_paragraph_break` for each chunk boundary (used for silence gap sizing)
4. No overlap — each chunk is independent

```python
import re

def split_into_sentences(text):
    raw_parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s.strip() for s in raw_parts if s.strip()]

def build_chunks(text, max_chars=1500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Build sentence list with paragraph break markers
    sentence_entries = []
    for i, para in enumerate(paragraphs):
        for j, sent in enumerate(split_into_sentences(para)):
            is_para_start = (j == 0 and i > 0)
            sentence_entries.append({"text": sent, "para_break_before": is_para_start})

    chunks = []
    current_sentences, current_len = [], 0
    has_paragraph_break = False
    for entry in sentence_entries:
        added_len = len(entry["text"]) + (1 if current_len > 0 else 0)
        if current_len + added_len > max_chars and current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "has_paragraph_break": has_paragraph_break,
            })
            current_sentences, current_len = [entry["text"]], len(entry["text"])
            has_paragraph_break = entry["para_break_before"]
        else:
            current_sentences.append(entry["text"])
            current_len += added_len
            if entry["para_break_before"]:
                has_paragraph_break = True
    if current_sentences:
        chunks.append({"text": " ".join(current_sentences), "has_paragraph_break": has_paragraph_break})

    return chunks
```

### Generation Parameters for Consistency
Lower temperature reduces stochastic voice drift between and within chunks:

```python
for result in model.generate(
    text=chunk["text"],
    voice=VOICE,
    temperature=0.4,    # Default 0.8 causes voice drift; 0.3-0.5 sweet spot
    top_k=30,           # Default 50; tighter for consistency
    top_p=0.90,         # Default 0.95; slightly tighter nucleus sampling
):
    ...
```

### Audio Stitching (v5) — Silence Gaps, Not Crossfade
- **No silence trimming** — model naturally generates pauses; do NOT strip them
- **300ms silence gap** between regular chunks (sentence boundaries)
- **700ms silence gap** at paragraph boundaries (`has_paragraph_break = True`)
- **No crossfade** — speech needs clean gaps between segments, not blending
- Run `gc.collect()` every 5 chunks to keep memory in check

```python
def create_silence(duration_ms, sr=24000):
    """Create a silence array of the given duration."""
    return np.zeros(int(sr * duration_ms / 1000), dtype=np.float32)

def assemble_with_silence_gaps(chunk_audios, chunks, sr=24000):
    """Assemble chunk audio with appropriate silence gaps between them."""
    if not chunk_audios:
        return np.array([], dtype=np.float32)

    parts = [chunk_audios[0]]
    for i in range(1, len(chunk_audios)):
        # Use 700ms gap at paragraph breaks, 300ms otherwise
        gap_ms = 700 if chunks[i - 1]["has_paragraph_break"] else 300
        parts.append(create_silence(gap_ms, sr))
        parts.append(chunk_audios[i])

    return np.concatenate(parts)
```

### Complete v5.1 Pipeline
```
Text → Split sentences → Build chunks (1500 chars max, sentence boundaries)
  → For each chunk: Generate (temp=0.4, top_k=30) → Per-chunk RMS normalize
  → Assemble with silence gaps (300ms sentence / 700ms paragraph) → Noise reduce (0.3) → Final RMS normalize → WAV
```

## Post-Processing Pipeline (Critical for Long Text)

Raw model output often contains background noise artifacts (wind, hiss), especially with quantized models or long sequences. The v5 pipeline uses minimal post-processing — the model generates good audio natively, and over-processing destroys quality.

### 1. Spectral Gating Noise Reduction (Light)
Apply after assembling the full audio — removes stationary background noise (wind, hiss). Use a **lower strength (0.3)** than previous versions to preserve natural audio quality:

```python
import noisereduce as nr

def reduce_noise(audio, sample_rate=24000, strength=0.3):
    reduced = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=True,          # Best for consistent background noise
        prop_decrease=strength,    # 0.3 for v5 — lighter than v4's 0.6 to preserve natural sound
        n_fft=1024,               # Frequency resolution for 24kHz speech
        freq_mask_smooth_hz=200,  # Smooth frequency mask to avoid artifacts
        time_mask_smooth_ms=100,  # Smooth time mask to avoid choppy audio
    )
    return reduced.astype(np.float32)
```

- **strength 0.3** is the v5 sweet spot — just enough to clean up artifacts without degrading speech
- **stationary=True** — best for TTS artifacts which are consistent background noise
- Apply to the full assembled audio, not per-chunk (better noise profile estimation)
- Requires `pip install noisereduce`
- **Per-chunk RMS normalization restored in v5.1** — v5 removed it but v5.1 brings it back to fix volume decay within long chunks. Each chunk is normalized to target RMS before stitching, then final global RMS is applied after assembly.

### 2. Final Global RMS Normalization + Clipping
After noise reduction, apply a single RMS normalization pass to the complete audio, then clip to [-1.0, 1.0] for safety:

```python
def rms_normalize(audio, target_rms=0.08):
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        gain = min(target_rms / current_rms, 10.0)  # Cap extreme gains
        normalized = audio * gain
        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)  # Safety clip
        return normalized
    return audio
```

### Complete v5.1 Post-Processing Order
```python
# 1. Generate chunk audio (no overlap, no silence trimming)
# 1b. Per-chunk RMS normalization (fixes volume decay)
# 2. Assemble chunks with silence gaps (300ms sentence / 700ms paragraph)
audio_full = assemble_with_silence_gaps(chunk_audios, chunks)

# 3. Noise reduction on full audio (light — 0.3 strength)
audio_full = reduce_noise(audio_full)

# 4. Final RMS normalization + safety clip
audio_full = rms_normalize(audio_full)
audio_full = np.clip(audio_full, -1.0, 1.0)

# 5. Save
sf.write("output.wav", audio_full, 24000)
```

## Quantization Artifacts Warning

The 4-bit quantized model (`mlx-4bit`) can produce intermittent wind/hiss background noise, especially on long text. This is a known issue with Voxtral's architecture:

- **Root cause**: Voxtral has 3 stages — LLM backbone (3.4B), acoustic transformer (390M), codec decoder (300M). The acoustic transformer uses flow-matching which requires full precision. The codec decoder has audio-critical convolutions that degrade under quantization.
- **Naive 4-bit quantization produces garbage audio** — research from voxtral-int4 project confirms this
- **Safe quantization**: Only the LLM backbone should be quantized. Acoustic transformer and codec decoder must stay BF16.
- **6-bit model solves this**: `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` quantizes only the LLM backbone to 6-bit while keeping acoustic transformer and codec decoder at full BF16 precision — no wind/noise artifacts, ~2-3x faster than bf16.
- **Recommendation**: Use `mlx-6bit` for the best speed/quality balance (~1.33x real-time, clean audio). Use `mlx-bf16` only when maximum quality is critical and speed doesn't matter. Avoid `mlx-4bit` — noise artifacts are unavoidable.

## SSL Certificate Issue (Walmart Network)

Model download and pip install may fail with `SSL: CERTIFICATE_VERIFY_FAILED` on corporate network. Fix by setting environment variables before running:

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem
export REQUESTS_CA_BUNDLE=/etc/ssl/cert.pem
```

Or for pip installs specifically:
```bash
pip install <package> --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
```

## Performance Characteristics (Apple Silicon)

| Metric | 6-bit (Recommended) | bf16 (Max Quality) | 4-bit (Deprecated) |
|--------|---------------------|--------------------|--------------------|
| Model size | ~3.5GB | ~8GB | ~2.5GB |
| Generation speed | **~1.33x real-time** (M4 Pro) | ~0.4-0.5x real-time | Fastest |
| Memory footprint | ~4GB during inference | ~8GB during inference | ~3GB during inference |
| Audio quality | Near-bf16, clean | Best | Wind/hiss artifacts |
| Noise artifacts | ✅ None | ✅ None | ❌ Present |

| Metric | Value |
|--------|-------|
| Output sample rate | 24000 Hz |
| Output format | float32 PCM |
| Model load time | ~2-3s (cached), first run downloads model from HuggingFace |

## Known Gotchas

1. **tiktoken is mandatory** — `mlx-audio` does not list it as a hard dependency, but Voxtral's Tekken tokenizer requires it. Without it: model loads without error but generation crashes with `RuntimeError: Tokenizer not loaded. Ensure post_load_hook was called.`
2. **Voice parameter is a string** — must exactly match preset names (case-sensitive). Invalid voice names may silently fall back to a default or error.
3. **`model.generate()` is a generator** — you must iterate over it. Calling it without a loop gives you nothing.
4. **`result.audio` is `mx.array`** — not numpy. Always wrap with `np.array()` before soundfile or numpy operations.
5. **No streaming to file** — all audio is generated in memory. For very long texts (30K+ chars), total audio array can be large (~100MB+ for 10+ minutes of audio at 24kHz float32).
6. **Apple Silicon only** — MLX does not run on Intel Macs or Linux/Windows. For non-Mac, use vLLM-Omni with CUDA GPU instead.
7. **Non-commercial license** — CC BY-NC 4.0. Production/commercial use requires Mistral API.
8. **4-bit model produces wind/noise artifacts** — The quantized acoustic transformer and codec decoder lose precision, causing intermittent background noise. Use 6-bit or bf16 model for clean output. 6-bit only quantizes the LLM backbone and keeps audio-critical components at BF16.
9. **6-bit model requires ~3.5GB free disk space** — If disk is full, delete unused cached models from `~/.cache/huggingface/hub/` (e.g., remove 4-bit model if no longer needed).
10. **noisereduce too aggressive = robotic speech** — v5 uses `prop_decrease=0.3` (lighter touch). Going above 0.5 risks removing frequency content and making speech sound artificial. Previous versions used 0.6 which was too aggressive.
11. **Each model.generate() creates a fresh KV cache** — There is NO context carryover between calls. Every chunk starts from zero. Use 1500-char chunks as the sweet spot — fewer boundaries than 300 (voice drift) but shorter than 2500 (volume decay).
12. **Default temperature (0.8) causes voice drift** — For narration/long text, use temperature 0.3-0.5, top_k 30, top_p 0.90. The default is designed for creative variety, not consistent narration.
13. **Do NOT trim silence from chunk audio** — The model naturally generates pauses between sentences. Previous versions aggressively trimmed silence which destroyed natural speech rhythm. Let the model's native pauses through; they make the output sound human.
14. **Use silence-gap stitching, not crossfade** — Speech needs clean gaps between segments. Crossfading blends the end of one sentence into the start of another, creating unnatural transitions. Use 300ms silence between sentence chunks and 700ms at paragraph boundaries.
15. **Volume decays within long chunks (>45s)** — The LLM's attention to the voice embedding dilutes as the KV cache grows. At 2500-char chunks (~60-90s audio), the end of each chunk is noticeably quieter than the start. Fix: use 1500-char chunks (~30-45s) and apply per-chunk RMS normalization to level out the volume before stitching.
16. **Aggressive noise reduction amplifies volume decay** — At `prop_decrease` > 0.5, the spectral gating strips more audio from the already-quiet end-of-chunk sections, making the fade-out worse. Keep noise reduction at 0.3 max when using long chunks.
