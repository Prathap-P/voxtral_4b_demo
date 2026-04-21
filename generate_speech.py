"""
Voxtral-4B-TTS v4 — Long Text Optimized Generator (Quality Mode)
Fixes chunk boundary word truncation and voice drift via:
  - Text overlap conditioning (last sentence carried forward between chunks)
  - Lower sampling temperature for voice consistency
  - Sentence-only boundary splitting (no mid-sentence cuts)
  - Silence trimming + raised-cosine crossfade + RMS loudness matching
  - Spectral gating noise reduction

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
MAX_CHUNK_CHARS = 300            # Shorter chunks = less voice drift within each chunk
CROSSFADE_MS = 120               # Raised-cosine crossfade overlap at boundaries

# Generation parameters — lower temp = more consistent voice across chunks
TEMPERATURE = 0.4                # Default is 0.8; lower = less stochastic voice drift
TOP_K = 30                       # Default is 50; tighter sampling for consistency
TOP_P = 0.90                     # Default is 0.95; slightly tighter nucleus sampling

# Post-processing
ENABLE_NOISE_REDUCTION = True    # Set False to compare raw vs cleaned output
NOISE_REDUCE_STRENGTH = 0.6     # 0.0 = no reduction, 1.0 = aggressive (0.5-0.7 is sweet spot)

# Silence trimming
SILENCE_THRESHOLD_DB = -40       # RMS threshold below which audio is considered silence
SILENCE_MIN_DURATION_MS = 50     # Don't trim silence shorter than this

# ─── Text ────────────────────────────────────────────────────────────────────
# Replace this with your own text (supports 20K-30K+ characters)
TEXT = (
    """Concerns are rising about other nations using AI to remove humans from decision-making. The Pentagon moved to ban certain providers like Anthropic earlier this year... due to supply chain risks. This prompted a discussion with Under Secretary Emil Michael about how AI might change warfare... He draws an analogy to the ride-sharing industry. Safety statistics for systems like Tesla FSD are actually amazing. The fear is of change itself... in reality, the technology makes service more reliable and precise. Much like how Uber reduced drinking driving while increasing availability... applying this to military contexts means being able to discern a decoy from a non-decoy within drone swarms. A recent demonstration by Cameron Stanley... showcased this through a program called Maven Smart System. Specifically, the Target Workbench allows users to view live images and select targets within a unified workflow. It is not about replacing human judgment... but increasing the "human context window" by synthesizing vast amounts of data.

When a target is identified, the system calculates variables such as weather conditions... fuel consumption, and collateral effects. It does not operate like a chatbot or Skynet; instead, it serves as an orchestration layer on top of data streams. The Under Secretary clarifies that no Large Language Model is baked into the kill chain itself... countering common misconceptions about automated killing. Instead, tools like Palantir surface choices that are otherwise consumed by spreadsheets and PowerPoint files—methods historically used to relay target lists. The digitalization of targeting processes accelerates these decisions, granting a single operator the power of many more. While permissions and authorities remain strictly human-controlled to ensure checks and balances, the system provides better outcomes through informed clicks. This shift from manual coordination to AI-assisted synthesis represents a responsible evolution of war fighting... moving beyond the chaos of unconnected data to a unified strategy.

The discussion outlines three layers of artificial intelligence application within defense, starting with efficiency... Mundane work is streamlined so personnel can focus on more interesting tasks. Then there is the intelligence layer... Imagine all the intelligence gathered from satellite imagery worldwide. Currently, a human analyst must look at everything to make a judgment... but with historical data and AI synthesis, the system can identify anomalies. It learns what those anomalies are—creating a totally different paradigm for intelligence analysis if you will. Moving on to the third layer, war fighting... AI takes all paperwork and modeling and simulation to react faster. But also more precisely.

These are tangible ways AI is used, yet there remains skepticism about its role in decision-making. Speed wins the game... Look at what happened in Venezuela—the speed at which that execution of that operation meant there were no casualties on our side. If you had to spend way more time, you weren't able to synthesize information as well... Speed has to be one of our prerogatives, but better information is the goal so that decisions are more precise. Is there a limit to what this can do? The interviewee confirms there is... No one believes there is some all-seeing, all-knowing answer to human conflict which has been happening since humans existed. Ultimately what you want is clear objectives, manpower and machinery to do it... with the least cost, the least amount of damage, and quickest time.

The worry is about other countries who don't have that... They use AI to take humans out of the decision-making process because they distrust their generals due to graft. Which governments are referenced? China represents the biggest military buildup in world history, and there has been a purge of generals... How do you replace all these people? What is the command and control? It is just a different mindset. Currently, LLMs are largely chatbot uses... but the AI industry is moving towards agents—letting the AI take some action for you. The plan here is not to automate warfare... Yet if an adversary does that, can you really afford to sit still and do it by the book? It becomes tempting when an LM gets 99% of the way there... The response is that the US must be AI dominant so we are never faced where counterforce AI is better than our AI.

People confuse automation with an automated army... What about an automated mind sweeping or detection operation? There is no human underwater... yet there is an action you want to take. Well, we don't want mines on our shores... Or there is a missile coming at you and you want to take it down from space. Like Golden Dome, how do you do that? You have to do that in 90 seconds when it is launched. Those kinds of things are where automation fits, but human oversight remains on the most consequential decisions.

The conversation opens with the critical challenge of retrieving systems from space within a ninety-second window following launch. While extreme circumstances require automation capabilities, mobilizing an entire fleet or army remains beyond current operational planning... Consequently, a thirty-five-page Department of Defense directive mandates human oversight to ensure controls are constantly updated and systems are managed properly. Moving on, data layers offer another potential utility... specifically regarding strikes before they occur. Consider the school in Minab, Iran, where playground markings might signal a target to avoid firing at civilians. This is the point of using autonomous systems... they augment human decision-making rather than replacing it. It could operate on the front end to flag warnings or the back end to verify targets, but ultimately humans must make the final decision.

Regarding terminology... Large Language Models are evolving beyond text-only processing. They will become visual models trained on robotics, consumer data like Nest Cams and YouTube movement... These proprietary datasets are becoming incredibly valuable for training AI. Let us turn to drone warfare, specifically the contrasting scenarios in Ukraine and Iran... In Russia and Ukraine... there is a battle over territory where lines are drawn. Here, robots act as the front line while humans remain back... sending machines first to reduce human risk. Conversely, in Iran, the lesson is about cost imbalance. A cheap drone can threaten expensive targets... forcing defenders to use multi-million dollar counter measures against low-cost threats.

This has driven a push for mass attritable weapons... affordable systems that can be manufactured quickly and designed to be lost. These differ from billion-dollar platforms taking ten years to build... The new drone dominance program focuses on low-cost units around thirty thousand dollars each. Acting as one-way attack drones similar to Shahid models, while Ukraine offered collaboration... the program prioritizes supply chains free from adversary dependency. Ensuring onshore manufacturing where possible, China has been displaying drone swarms that look like art but function as military simulations... armed drones communicating to reform against defenses. Defending small bases or garrisons is a new challenge that emerged from the Ukraine Russia war... requiring both offensive and defensive strategies. A counter unmanned systems task force is now looking at lasers, directed energy... and electronic warfare to take down these interoperable threats. Finally, cyber warfare faces similar AI impacts... models trained on code learn vulnerabilities quickly. This presents risk and opportunity as adversaries distill frontier model capabilities for the next wave of innovation... making AI critical to Pentagon operations regarding targeting, information synthesis, and security fronts. The discussion will now shift to selecting AI vendors... focusing on the situation with Anthropic and other key topics.

Now, a recent Big Technology Podcast segment featured host Emil Michael. They discussed the Under Secretary's stance on AI vendors... OpenAI focused on consumers, but Anthropic targeted government services. This happened after Biden's executive order... Google is catching up with specific divisions yet a culture clash remains. The Department of War lacks safeguards in the public eye... But decades of procedures exist internally.

So, the Maven Smart System contract was renegotiated under scrutiny... Anthropic desired provisions against mass surveillance. Policy banned AI for kinetic actions. It took three months to explain why exceptions can't run a department of three million people... The contract was called off. Pentagon deemed Anthropic a supply chain risk. Michael questioned how they were banned if they agreed to lawful uses...

Now, regarding cyber capabilities, Michael highlighted Mythos and project Glass Wing. An AI security institute assessment found it completed a thirty-two step corporate network attack end-to-end... Human experts would take twenty hours to replicate this autonomous cyber weaponization. While not encouraging use, he argued for keeping the tool against drone threats. Choosing one provider is an original sin... requiring gargantuan software effort on classified networks. Adversaries use distillation attacks showing up in Deep Seek within months...

The dialogue likened AI to a cyber nuclear bomb due to potential outcomes. Concerns include forty percent unemployment... or global coercion alongside bio-chemical risks. Management teams are judged on national security fit, distinct from foreign chip manufacturers... Hosting Anthropic models on Amazon cloud requires control over upgrade cycles compressing to three months. These updates introduce bugs alongside new model weights and guardrails... Scientists at the CDC assumed a refusal was from bad actors. Yet it was an off-the-shelf model...

A culture clash exists between Silicon Valley and the government, referencing Pete Hegseth's stance. No company should dictate terms to the Department of Defense... Despite revenue tripling in three months, a legal challenge involved Judge Rita Lynn ruling records showed hostility through the press... The speaker dismissed this as overreaching because vendors can't dictate government hiring based on disagreement. Ultimately, the Department of Defense cares about war fighters and national security above all else...

With three million employees, they require trust to handle unpredictability in Iran or similar conflicts. Google Gemini faced protests but returned working with the Pentagon, proving alignment is critical during actual conflicts... The government prioritizes safety and mission success over vendor convenience regarding the 2018 Maven project. The official expressed hope that tech companies mature in understanding government partnerships now...

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


# ─── Chunking with Text Overlap ─────────────────────────────────────────────
def build_chunks_with_overlap(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """
    Build chunks with text overlap for context conditioning.

    Each chunk includes:
      - 'overlap_text': Last sentence(s) from previous chunk (context, audio will be trimmed)
      - 'new_text': New sentences to generate audio for
      - 'full_text': overlap_text + new_text (what gets sent to the model)

    The overlap gives the model prosodic context from the previous chunk,
    solving word truncation at boundaries and voice drift between chunks.
    """
    # First, flatten paragraphs and split into sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    all_sentences = []
    paragraph_break_indices = set()  # Track where paragraph breaks occur for natural pauses

    for para in paragraphs:
        if all_sentences:
            paragraph_break_indices.add(len(all_sentences))
        sentences = split_into_sentences(para)
        all_sentences.extend(sentences)

    if not all_sentences:
        return [{"overlap_text": "", "new_text": text, "full_text": text}]

    chunks = []
    current_new_sentences = []
    current_len = 0

    for i, sentence in enumerate(all_sentences):
        # Would adding this sentence exceed the limit?
        added_len = len(sentence) + (1 if current_len > 0 else 0)

        if current_len + added_len > max_chars and current_new_sentences:
            # Flush current chunk
            new_text = " ".join(current_new_sentences)

            # Overlap: last sentence from this chunk becomes context for the next
            overlap_sentence = current_new_sentences[-1]

            # Build the previous chunk's overlap context
            if chunks:
                # This chunk gets overlap from the previous chunk
                pass  # already handled when chunk was created

            chunks.append({
                "new_text": new_text,
                "overlap_for_next": overlap_sentence,
                "has_paragraph_break": i in paragraph_break_indices,
            })

            current_new_sentences = [sentence]
            current_len = len(sentence)
        else:
            current_new_sentences.append(sentence)
            current_len += added_len

    # Flush remaining sentences
    if current_new_sentences:
        new_text = " ".join(current_new_sentences)
        chunks.append({
            "new_text": new_text,
            "overlap_for_next": current_new_sentences[-1],
            "has_paragraph_break": False,
        })

    # Now build final chunks with overlap context prepended
    final_chunks = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            # First chunk: no overlap
            overlap_text = ""
            full_text = chunk["new_text"]
        else:
            # Get overlap from previous chunk
            overlap_text = chunks[idx - 1]["overlap_for_next"]
            full_text = overlap_text + " " + chunk["new_text"]

        final_chunks.append({
            "overlap_text": overlap_text,
            "new_text": chunk["new_text"],
            "full_text": full_text,
            "has_paragraph_break": chunk["has_paragraph_break"],
        })

    return final_chunks


# ─── Audio Processing Utilities ──────────────────────────────────────────────
def trim_silence(audio: np.ndarray, threshold_db: float = SILENCE_THRESHOLD_DB,
                 min_dur_ms: float = SILENCE_MIN_DURATION_MS) -> np.ndarray:
    """
    Trim leading and trailing silence from audio using RMS energy detection.
    Preserves a small buffer to avoid cutting into speech onset.
    """
    if len(audio) == 0:
        return audio

    # Calculate RMS energy in small windows
    window_samples = int(SAMPLE_RATE * 0.01)  # 10ms windows
    threshold_linear = 10 ** (threshold_db / 20.0)
    min_samples = int(SAMPLE_RATE * min_dur_ms / 1000)

    # Find first non-silent sample
    start = 0
    for pos in range(0, len(audio) - window_samples, window_samples):
        window = audio[pos:pos + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms > threshold_linear:
            start = max(0, pos - min_samples)  # Keep small buffer before speech
            break

    # Find last non-silent sample
    end = len(audio)
    for pos in range(len(audio) - window_samples, start, -window_samples):
        window = audio[pos:pos + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms > threshold_linear:
            end = min(len(audio), pos + window_samples + min_samples)
            break

    return audio[start:end]


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


def cosine_crossfade(seg1: np.ndarray, seg2: np.ndarray,
                     overlap_ms: int = CROSSFADE_MS) -> np.ndarray:
    """
    Crossfade two audio segments using a raised cosine (Hann) window.
    Smoother than linear crossfade — no energy dip at the midpoint.
    """
    overlap_samples = int(SAMPLE_RATE * overlap_ms / 1000)
    if len(seg1) < overlap_samples or len(seg2) < overlap_samples:
        return np.concatenate([seg1, seg2])

    # Raised cosine (Hann) windows — smoother than linear
    t = np.linspace(0, np.pi, overlap_samples, dtype=np.float32)
    fade_out = (1 + np.cos(t)) / 2      # 1 → 0 smoothly
    fade_in = (1 - np.cos(t)) / 2       # 0 → 1 smoothly

    seg1_end = seg1[-overlap_samples:] * fade_out
    seg2_start = seg2[:overlap_samples] * fade_in
    blended = seg1_end + seg2_start

    return np.concatenate([seg1[:-overlap_samples], blended, seg2[overlap_samples:]])


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
    print("  Voxtral-4B-TTS v4 — Long Text Optimized (Quality Mode)")
    print("  Fixes: chunk boundary truncation, voice drift, consistency")
    print("=" * 70)

    # Build chunks with overlap context
    chunks = build_chunks_with_overlap(TEXT)
    total_new_chars = sum(len(c["new_text"]) for c in chunks)

    print(f"\n  Model:          {MODEL_ID}")
    print(f"  Voice:          {VOICE}")
    print(f"  Temperature:    {TEMPERATURE} (lower = more consistent voice)")
    print(f"  Top-k / Top-p:  {TOP_K} / {TOP_P}")
    print(f"  Text:           {len(TEXT):,} characters")
    print(f"  Chunks:         {len(chunks)} (max {MAX_CHUNK_CHARS} chars, sentence boundaries)")
    print(f"  Overlap:        Last sentence carried forward between chunks")
    print(f"  Crossfade:      {CROSSFADE_MS}ms raised-cosine")
    print(f"  Noise reduce:   {'ON (strength={})'.format(NOISE_REDUCE_STRENGTH) if ENABLE_NOISE_REDUCTION else 'OFF'}")
    print(f"  Normalization:  RMS loudness matching")
    print(f"  Output:         {OUTPUT_FILE}")
    print()

    # Load model
    print("Loading model (bf16 = ~8GB, first run downloads from HuggingFace)...")
    start_load = time.time()
    model = load(MODEL_ID)
    load_time = time.time() - start_load
    print(f"Model loaded in {format_time(load_time)}")

    # Generate speech chunk by chunk with overlap conditioning
    print(f"\nGenerating speech ({len(chunks)} chunks with overlap conditioning)...")
    print("-" * 70)

    start_gen = time.time()
    all_audio_segments = []
    chars_done = 0

    for i, chunk_info in enumerate(chunks):
        chunk_start = time.time()
        overlap_text = chunk_info["overlap_text"]
        new_text = chunk_info["new_text"]
        full_text = chunk_info["full_text"]
        has_para_break = chunk_info["has_paragraph_break"]

        chars_done += len(new_text)
        progress_pct = (chars_done / total_new_chars) * 100

        # Show progress
        preview = new_text[:55].replace("\n", " ")
        if len(new_text) > 55:
            preview += "..."
        overlap_info = f" [+{len(overlap_text)}ch overlap]" if overlap_text else ""
        print(f"\n  [{i+1}/{len(chunks)}] ({progress_pct:5.1f}%) \"{preview}\"{overlap_info}")
        sys.stdout.flush()

        # Generate audio for full text (overlap + new) with controlled sampling
        chunk_audio_parts = []
        for result in model.generate(
            text=full_text,
            voice=VOICE,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
        ):
            chunk_audio_parts.append(np.array(result.audio))

        if not chunk_audio_parts:
            print(f"           WARNING: No audio generated for chunk {i+1}")
            continue

        full_audio = np.concatenate(chunk_audio_parts) if len(chunk_audio_parts) > 1 else chunk_audio_parts[0]

        # Trim the overlap audio (proportional estimation)
        if overlap_text and len(full_text) > 0:
            overlap_ratio = len(overlap_text) / len(full_text)
            trim_samples = int(len(full_audio) * overlap_ratio)
            chunk_audio = full_audio[trim_samples:]
        else:
            chunk_audio = full_audio

        # Trim silence from edges (model sometimes adds leading/trailing silence)
        chunk_audio = trim_silence(chunk_audio)

        if len(chunk_audio) == 0:
            print(f"           WARNING: Chunk {i+1} produced empty audio after trimming")
            continue

        # RMS normalize for consistent perceived loudness
        chunk_audio = rms_normalize(chunk_audio)

        chunk_duration = len(chunk_audio) / SAMPLE_RATE
        chunk_time = time.time() - chunk_start
        elapsed = time.time() - start_gen
        eta = (elapsed / chars_done) * (total_new_chars - chars_done) if chars_done > 0 else 0

        print(f"           Done: {format_time(chunk_duration)} audio | "
              f"took {format_time(chunk_time)} | "
              f"ETA: {format_time(eta)}")

        all_audio_segments.append(chunk_audio)

        # Periodic memory cleanup
        if (i + 1) % 10 == 0:
            gc.collect()

    gen_time = time.time() - start_gen

    if not all_audio_segments:
        print("\nERROR: No audio segments generated!")
        return

    # Assemble final audio with raised-cosine crossfades
    print(f"\nAssembling {len(all_audio_segments)} segments with cosine crossfade...")
    if len(all_audio_segments) == 1:
        audio_full = all_audio_segments[0]
    else:
        audio_full = all_audio_segments[0]
        for seg in all_audio_segments[1:]:
            audio_full = cosine_crossfade(audio_full, seg)

    # Post-processing: Noise Reduction
    if ENABLE_NOISE_REDUCTION:
        print(f"Applying noise reduction (strength={NOISE_REDUCE_STRENGTH})...")
        nr_start = time.time()
        audio_full = reduce_noise(audio_full)
        nr_time = time.time() - nr_start
        print(f"  Noise reduction done in {format_time(nr_time)}")

    # Final RMS normalization of assembled audio
    audio_full = rms_normalize(audio_full)

    # Safety: clip to prevent any floating-point overflow
    audio_full = np.clip(audio_full, -1.0, 1.0)

    # Save as WAV
    sf.write(OUTPUT_FILE, audio_full, SAMPLE_RATE)

    # ─── Results ─────────────────────────────────────────────────────────
    duration = len(audio_full) / SAMPLE_RATE
    file_size_mb = (len(audio_full) * 4) / (1024 * 1024)

    print(f"\n{'=' * 70}")
    print(f"  Generation Complete! (v4 — Overlap Conditioning)")
    print(f"{'=' * 70}")
    print(f"  File:            {OUTPUT_FILE}")
    print(f"  File size:       {file_size_mb:.1f} MB")
    print(f"  Audio duration:  {format_time(duration)}")
    print(f"  Generation time: {format_time(gen_time)}")
    print(f"  Real-time factor: {gen_time/duration:.2f}x (lower is faster)")
    print(f"  Sample rate:     {SAMPLE_RATE} Hz")
    print(f"  Chunks used:     {len(chunks)}")
    print(f"  Characters:      {total_new_chars:,}")
    print(f"  Temperature:     {TEMPERATURE}")
    print(f"  Noise reduction: {'ON' if ENABLE_NOISE_REDUCTION else 'OFF'}")
    print(f"  Overlap:         Sentence-level context conditioning")
    print(f"  Crossfade:       {CROSSFADE_MS}ms raised-cosine")
    print(f"{'=' * 70}")
    print(f"\nOpen '{OUTPUT_FILE}' in any audio player to listen!")
    print(f"Compare with previous version to hear the improvement.")


if __name__ == "__main__":
    main()
