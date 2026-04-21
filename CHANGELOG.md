# Voxtral TTS — Changes Log

## v5.1 — Volume Decay Fix

### Problem
Audio volume gradually fades within each chunk — loud at the start, quiet at the end. When the next chunk starts, volume jumps back up creating a "sawtooth" loudness pattern.

### Root Cause
The LLM backbone's attention to the voice embedding tokens dilutes as the KV cache grows during autoregressive generation. At 2500 chars (~60-90s audio, ~750-1125 frames), the voice conditioning signal decays noticeably. Additionally, aggressive noise reduction (0.8) was stripping more audio from the already-quiet end-of-chunk sections.

### What Changed

| # | Change | Before (v5) | After (v5.1) | Why |
|---|--------|-------------|-------------|-----|
| 1 | Chunk size | 2500 chars (~60-90s) | 1500 chars (~30-45s) | Shorter chunks = less time for attention to decay |
| 2 | Per-chunk RMS | Removed | Restored | Levels out volume fade within each chunk before stitching |
| 3 | Noise reduction | 0.3 (but user tested at 0.8) | 0.3 (enforced) | >0.5 amplifies the volume decay by stripping quiet passages |

---

## v5 — Less Processing, More Natural (Audio Quality Overhaul)

### Problems Solved
- **No natural pauses between sentences** — audio sounded like continuous run-on speech
- **Voice changes between chunks** — different person speaking across boundaries
- **Overall unnatural sound** — over-processing destroyed audio quality

### Root Cause (from deep analysis)
- `trim_silence()` stripped ALL natural pauses the model generates
- `has_paragraph_break` was tracked but NEVER used — paragraph breaks produced same gap as regular chunks
- Crossfade OVERLAPPED audio (ate 120ms) instead of adding silence gaps
- 300-char chunks created ~40+ boundaries = 40 chances for voice drift
- Proportional overlap trimming was only ~90% accurate, causing boundary artifacts
- Per-chunk RMS normalization + noise reduction at 0.6 degraded audio

### What Changed

| # | Change | Before (v4) | After (v5) | Why |
|---|--------|-------------|------------|-----|
| 1 | **Chunk size 300→2500 chars** | 300 chars max | 2500 chars max, sentence boundaries | ~5-6 boundaries instead of ~40, 85% fewer chances for voice drift |
| 2 | **Removed overlap conditioning** | Last sentence prepended as context; overlap audio trimmed | No overlap — chunks are independent | With 5-6 boundaries, overlap adds complexity for minimal benefit |
| 3 | **Removed trim_silence()** | Leading/trailing silence trimmed per chunk (RMS energy detection) | No silence trimming — raw model output preserved | Model naturally generates pauses; stripping them caused "no pauses" problem |
| 4 | **Replaced crossfade with silence gaps** | 120ms raised-cosine crossfade (Hann window) | 300ms silence between chunks, 700ms at paragraph breaks | Speech needs gaps not blending — crossfade was eating 120ms of audio |
| 5 | **Used has_paragraph_break** | Tracked but never used — same gap everywhere | Paragraph boundaries get 700ms silence vs 300ms for regular chunks | Natural speech has longer pauses between paragraphs than between sentences |
| 6 | **Reduced post-processing** | Noise reduction 0.6, per-chunk RMS normalization | Noise reduction 0.3, removed per-chunk RMS, only final global RMS | Less processing = more natural; model output is good enough without heavy processing |

### Processing Pipeline (v5)
```
Text → Split sentences → Build chunks (2500 chars max, sentence boundaries)
  → For each chunk: Generate (temp=0.4, top_k=30) — no overlap, no trimming
  → Assemble with silence gaps (300ms sentence / 700ms paragraph) → Noise reduce (0.3) → Final RMS normalize → WAV
```

### Key Insight
The v4 pipeline was over-engineered — trimming, crossfading, overlap conditioning, per-chunk normalization. The model generates good audio natively; the processing was destroying it.

---

## v4 — Long-Text Quality Fix (Overlap Conditioning + Voice Consistency)

### Problems Solved
- **Word truncation at chunk boundaries** — first word of new chunks getting cut off
- **Voice drift within chunks** — voice character changing between start and end of same chunk
- **Inconsistency across chunks** — voice timbre varying between adjacent chunks

### Root Cause (from source code analysis)
Each `model.generate()` call creates a **fresh KV cache** — every chunk starts from zero context with no knowledge of what came before. Combined with high sampling temperature (0.8 default), this causes stochastic voice drift both within and between chunks.

### What Changed

| # | Change | Before (v3) | After (v4) | Why |
|---|--------|-------------|------------|-----|
| 1 | **Text overlap conditioning** | Independent chunks, no context | Last sentence of previous chunk prepended as context; overlap audio trimmed | Model gets prosodic continuity — knows what "came before" for natural flow |
| 2 | **Lower sampling temperature** | Default 0.8 / top_k 50 / top_p 0.95 | 0.4 / top_k 30 / top_p 0.90 | Reduces stochastic voice variation between and within chunks |
| 3 | **Smaller chunks (300 chars)** | 500 chars | 300 chars, sentence boundaries only | Less time for autoregressive drift; voice embedding stays influential |
| 4 | **Sentence-only splitting** | Split at paragraphs → sentences → clauses (commas) | Split only at sentence boundaries (. ! ? ...) | No mid-sentence cuts = cleaner prosody at chunk edges |
| 5 | **Silence trimming** | Fixed 400ms silence gap between chunks | Trim leading/trailing silence per chunk (RMS energy detection) | Model naturally generates pauses; fixed gaps were doubling pauses |
| 6 | **Raised-cosine crossfade** | 50ms linear crossfade | 120ms raised-cosine (Hann window) | No energy dip at midpoint; smoother transitions |
| 7 | **RMS loudness normalization** | Peak normalization (0.9 target) | RMS normalization (0.08 target RMS) | Perceived loudness matching is more consistent than peak matching |

### Processing Pipeline (v4)
```
Text → Split sentences → Build chunks (300 chars max)
  → For each chunk: prepend overlap sentence → Generate (temp=0.4)
  → Trim overlap audio (proportional) → Trim silence → RMS normalize
  → Assemble with cosine crossfade (120ms) → Noise reduce → Final RMS normalize → WAV
```

### Key Insight
The Voxtral model internally uses `context_frames = 16` for streaming coherence within a single generation, but we were making N completely independent calls that bypassed this. Text overlap at the application level simulates the context the model needs.

---

## v3 — Fast Mode (6-bit) + Speed Optimization

### What Changed

| # | Change | Details | Why |
|---|--------|---------|-----|
| 1 | **Added `generate_speech_fast.py`** | New script using `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit` (~3.5GB) | 6-bit model is ~2-3x faster than bf16 while preserving acoustic transformer precision — no wind/noise artifacts. Benchmarked at 1.33x real-time on M4 Pro. |
| 2 | **Parallel to bf16 version** | Both scripts coexist — `generate_speech.py` (bf16, max quality) and `generate_speech_fast.py` (6-bit, fast) | Lets you A/B compare speed vs quality. Same text, same pipeline, different model. |
| 3 | **Separate output files** | bf16 → `output1.wav`, 6-bit → `output_fast.wav` | Easy side-by-side comparison without overwriting. |

### Model Comparison

| Variant | File | Size | Speed | Quality | Noise |
|---------|------|------|-------|---------|-------|
| 4-bit | *(deprecated)* | 2.5GB | Fastest | Wind/hiss artifacts | ❌ Bad |
| **6-bit** | `generate_speech_fast.py` | **3.5GB** | **~1.33x real-time** | **Near bf16** | ✅ Clean |
| bf16 | `generate_speech.py` | 8GB | Slowest | Best | ✅ Clean |

### Note
6-bit model requires ~3.5GB free disk space to download. If disk is full, delete unused cached models from `~/.cache/huggingface/hub/`.

---

## v2 — Noise Fix Update

### What Changed

| # | Change | Before | After                                                                         | Why                                                                                                                                                                                                           |
|---|--------|--------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | **Model switched to bf16** | `mlx-4bit` (2.5GB) | `mlx-bf16` (8GB) .<br/> fp16 is present in local folder ~/.cache/huggingface/hub/. | 4-bit quantization corrupts the acoustic transformer & codec decoder — these stages need full precision. This was the #1 cause of wind/hiss noise. |
| 2 | **Chunk size reduced** | 800 chars | 500 chars                                                                     | Shorter chunks = model stays in its "comfort zone" for prosody. Long sequences cause the autoregressive decoder to drift, introducing noise.                                                                  |
| 3 | **Noise reduction added** | None (raw output) | `noisereduce` spectral gating (strength 0.6)                                  | Removes any remaining stationary background noise (wind, hiss) via frequency-domain filtering.                                                                                                                |
| 4 | **Volume normalization added** | None | Per-chunk + final normalization                                               | Each chunk had slightly different volume levels, causing audible inconsistencies. Now normalized to 0.9 peak.                                                                                                 |
| 5 | **New dependency** | — | `noisereduce`                                                                 | Python library for spectral gating noise reduction.                                                                                                                                                           |

### Processing Pipeline Order
```
Text → Chunk (500 chars) → Generate → Normalize chunk → Silence gap → Crossfade → Assemble → Noise reduce → Final normalize → WAV
```

### Bottom Line
The wind noise was primarily caused by 4-bit quantization destroying precision in audio-critical model components. Switching to bf16 + adding post-processing noise cleanup eliminated it.
