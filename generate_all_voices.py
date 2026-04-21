"""
Voxtral-4B-TTS — Voice Comparison Demo
Generates the same short text in all 5 English voice presets.
Output: voice_samples/ directory with one WAV per voice.
"""

import os
import time
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
SAMPLE_RATE = 24000
OUTPUT_DIR = "voice_samples"

# All 5 English voice presets
VOICES = [
    "casual_male",
    "casual_female",
    "cheerful_female",
    "neutral_male",
    "neutral_female",
]

# Short conversational text — designed to reveal voice personality differences.
# Mix of statements, questions, emphasis, and natural pauses.
TEXT = (
    "You know what's funny? I spent the entire weekend trying to fix a bug "
    "that turned out to be a single missing comma. Honestly, I wasn't even mad... "
    "I just laughed and grabbed another coffee. Sometimes the simplest things "
    "trip you up the most, don't you think?"
)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  Voxtral-4B-TTS — Voice Comparison (5 English Presets)")
    print("=" * 65)
    print(f"\n  📦 Model:   {MODEL_ID}")
    print(f"  📝 Text:    {len(TEXT)} characters")
    print(f"  🎤 Voices:  {', '.join(VOICES)}")
    print(f"  💾 Output:  {OUTPUT_DIR}/")
    print(f"\n  \"{TEXT[:80]}...\"")
    print()

    # Load model once
    print("⏳ Loading model...")
    start_load = time.time()
    model = load(MODEL_ID)
    print(f"✅ Model loaded in {format_time(time.time() - start_load)}")
    print()

    # Generate for each voice
    results = []
    total_start = time.time()

    for i, voice in enumerate(VOICES):
        print(f"  [{i+1}/{len(VOICES)}] 🎤 Generating: {voice}...")
        gen_start = time.time()

        audio_parts = []
        for result in model.generate(text=TEXT, voice=voice):
            audio_parts.append(np.array(result.audio))

        audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
        gen_time = time.time() - gen_start
        duration = len(audio) / SAMPLE_RATE

        # Save WAV
        output_path = os.path.join(OUTPUT_DIR, f"{voice}.wav")
        sf.write(output_path, audio, SAMPLE_RATE)

        results.append({
            "voice": voice,
            "file": output_path,
            "duration": duration,
            "gen_time": gen_time,
        })

        print(f"         ✅ {format_time(duration)} audio | "
              f"took {format_time(gen_time)} | "
              f"saved → {output_path}")

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  ✅ All 5 Voices Generated!")
    print(f"{'=' * 65}")
    print(f"  ⏱️  Total time: {format_time(total_time)}")
    print()
    print(f"  {'Voice':<20} {'Duration':<12} {'File'}")
    print(f"  {'─' * 18}   {'─' * 10}   {'─' * 30}")
    for r in results:
        print(f"  {r['voice']:<20} {format_time(r['duration']):<12} {r['file']}")
    print(f"\n{'=' * 65}")
    print(f"\n🎧 Open the '{OUTPUT_DIR}/' folder and play each WAV to compare voices!")


if __name__ == "__main__":
    main()
