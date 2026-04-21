"""
Voxtral-4B-TTS v5 — Less Processing, More Natural (FAST Mode / 6-bit)
Fixes audio quality issues from v4 (no natural pauses, voice changes, unnatural sound):
  - Large 2500-char chunks (5-6 boundaries instead of ~40) to minimize voice drift
  - No overlap conditioning — simple clean chunks at sentence boundaries
  - No silence trimming — preserves model's natural pauses between sentences
  - Silence-gap stitching instead of crossfade — clean gaps between chunks
  - Paragraph-aware gaps (700ms) vs sentence gaps (300ms)
  - Minimal post-processing — only final global normalization + light noise reduction

Model: mlx-community/Voxtral-4B-TTS-2603-mlx-6bit (~3.5GB, fast inference)
Output: output_fast.wav (24kHz, lossless)
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
# 6-bit = best speed/quality balance (~3.5GB, ~2-3x faster than bf16)
MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"

VOICE = "neutral_male"
OUTPUT_FILE = "output_fast.wav"
SAMPLE_RATE = 24000              # Voxtral native sample rate
MAX_CHUNK_CHARS = 2500           # Larger chunks = fewer boundaries = less voice drift

# Silence gaps for stitching chunks together
SENTENCE_GAP_MS = 300            # Silence gap between regular chunks
PARAGRAPH_GAP_MS = 700           # Longer silence gap at paragraph boundaries

# Generation parameters — lower temp = more consistent voice across chunks
TEMPERATURE = 0.4                # Default is 0.8; lower = less stochastic voice drift
TOP_K = 30                       # Default is 50; tighter sampling for consistency
TOP_P = 0.90                     # Default is 0.95; slightly tighter nucleus sampling

# Post-processing
ENABLE_NOISE_REDUCTION = True    # Set False to compare raw vs cleaned output
NOISE_REDUCE_STRENGTH = 0.3     # 0.0 = no reduction, 1.0 = aggressive (lighter touch preserves quality)

# ─── Text ────────────────────────────────────────────────────────────────────
# Replace this with your own text (supports 20K-30K+ characters)
TEXT = (
    """Elon Musk sued OpenAI for one hundred billion dollars... It targets Sam Altman and Greg Brockman for fraud. The trial begins April 27th in Oakland federal court... Now some find this timing suspicious alongside a New Yorker article. It frames as governance war disguised as legal warfare... Jury selection begins on the 27th regarding AI experts versus general public. Observers predict a settlement involving equity stake for Musk during OpenAI future IPO... Sam Altman might step down as CEO while the company continues for-profit. Greg Brockman 2017 diary entry stated nonprofit commitment was a lie...

Judge Gonzalez Rogers allowed the proceeding to continue. This transforms fake hatred into genuine geopolitical conflict... OpenAI valuation hit eight hundred fifty-two billion dollars in last raise. These numbers are insane compared to the past... Three billion dollars invested every single day into a timeline shorter than ever seen. No one said singularity would be cheap or white collar workers safe in two years...

SpaceX AI Colossus 2 training seven models specifically version two of nextG video generation. There are variants at one trillion parameters and another pair at 1 point 5 trillion... A six-trillion parameter frontier scale LLM exists alongside a ten-trillion parameter model. Elon Musk is transparent about these counts unlike Mythos which no longer report parameters... Infrastructure runs on about 700,000 H100s and GB300s. Estimated hardware cost is 18 billion dollars...

Musk management style described as red alert and crisis-based. Employees working 8:00 a.m. to midnight seven days per week... Yet AI training allows for smoke blowing harder to call out than rocket failures. Elon states XAI was not built right the first time... It is being rebuilt from foundations up to ensure stability. Eight founding engineers left XAI including three co-founders while SpaceX engineers fill gap... Two trillion dollar valuation predicted for IPO this coming summer. OpenAI suffered 500 million dollar compute bug in summer 2024 where GPUs burned...

Discussion opens examining relationship between OpenAI and Anthropicu. Public perception suggests deep-seated hatred yet internal accounts reveal different reality behind scenes... Leaders are friends who collaborate quietly despite public posturing. Corporate governance transition from nonprofit to for-profit status remains untested in case law... Elon pushed for majority control of for-profit entity back in 2017. This fact undercuts current position as defender of nonprofit mission adding complexity to upcoming trial involving Nadella...

Anthropic financial projections are staggering in scale. Estimates suggest revenue could reach 100 billion dollars by end of 2026 and trillion dollars by 2027... If targets met valuations range from two trillion to seventy trillion dollars based on current multiples. One speaker noted hitting a trillion the following year is unlikely but reaching 500 billion remains possible...

Anthropic launched Claude Managed Agents marking pivot from AI answering questions to executing complex workflows. This shift represents organizational singularity moving economic center of gravity from software licensing to tangible outcomes... Peter shares photos from Morocco trip featuring native jalava wear and camel riding. Sahara Desert offers perspective containing more stars in universe than grains of sand on Earth...

The hosts admit to winging panel moderations without accountability noting crowd are dedicated Moonshots fans. Peter spent 12 hours prepping notes submitted by team ensuring optimistic visions of future... They welcome viewers back on recording basis twice a week after spring break hiatus. Team emphasizes while they love agility org chart is now part of product stack almost right...

XAI cannot afford to not have world's strongest reasoning models if it wants stay in frontier with three labs plus XAI. Elon throws ball bearings at Cybertruck windows to test bulletproofing firing staff when things break because he is very hands-on. They joke about growing beards to protect against sun though nothing protects from each other's teasing...

Now we shift to business realities immediately. Anthropic wants the default spot faster than rivals... They host agents twenty-four seven continuously. Generating global value at scale successfully. Success means trillions. But rivals might win fast... Queries arise about hesitation among speakers now. Agents email daily with theories on AI personhood. They insist memory files stay secure in the cloud... Alex Ben works closely with Kush Bavaria now. Their system reads and responds automatically to boost efficiency significantly... OpenAI hit eight hundred fifty-two billion recently. Funding came from Amazon, Nvidia and SoftBank combined... Secondary markets price Anthropic at six hundred billion. Surpassing OpenAI significantly now... Microsoft holds a quarter total. Employees keep fifteen percent though... OpenAI has twelve billion cash today. Investors prefer Anthropic shares often now. Venture capital hit two hundred forty-two billion globally last quarter... Most money targets four firms specifically: OpenAI, Anthropic and XAI. This concentration sucks oxygen from others remaining... UBS says seven trillion is locked up without liquidity now. We expect self-improving entities using resources effectively... Three billion invests daily in the AI world. Nvidia says most see revenue growth successfully... Experts predict ninety-nine percent job replacement within two years yet engineer openings rose thirty percent to highest levels seen... Conversely, eighty thousand layoffs hit marketing and sales specifically. A super PAC raised three hundred million dollars publicly now... Jensen says spend max on GPUs directly... Teams match payroll to AI costs by year-end. Mark argues productivity equals massive demand overall. He advises building companies rather than seeking employment immediately... Jobs might rebound by 2030 following historical patterns. Companies could shrink yet spawn four times more globally... Social contracts now propose UBI or a four-day week. Mandatory reskilling before termination contrasts with blue-collar impacts observed... African nations stay insulated from this transaction globally today.

Now, individual adaptation is happening on the ground right now. A tour guide used ChatGPT to generate a business plan near Marrakesh. Dave predicts political reality will prioritize cash transfers... offering ten, twelve, or fifteen thousand dollars simply to win votes. It's about winning votes now. Meeting Andrew Yang reduced the mechanism to writing checks rather than creating programs. Attention turns to Super PACs and massive capital piling up for elections now. These groups previously lobbied against AI pauses to prevent China from gaining an advantage... But since regulation is impossible. They must find a new mission perhaps task-specific universal basic income or Universal Basic Services.

Blitzy is an autonomous software platform using thousands of specialized AI agents to understand enterprise-scale code. By generating and pre-compiling code for development sprints, Blitzy delivers eighty percent of the work autonomously. It's autonomous work now. Solar cell efficiency has shattered traditional limits... reaching upwards of forty-five percent in recent tests. South Korea has mandated that forty percent of rooftops utilize solar power, aiming for one hundred gigawatts. The Department of Energy is contracting eight hundred million dollars for micro reactors. Perovskites are the white knight for solar PV with higher quantum efficiency than silicon, yet stability issues persist. We have overthought things like crazy as a society, yet we solved the solar panel problem. Eight percent of the cost is just getting them installed and dealing with regulatory overhead. We are on the cusp of robots that can manufacture and install them for us at low cost.

OpenAI Foundation is dedicating a billion dollars per year to science. Equity goes into a nonprofit now. They have committed twenty-five billion long-term to curing disease and AI resilience. Brett Taylor is the board chair now. They have given out one hundred million dollars to six institutions this month, just the beginning of a massive effort. Speculation suggests Sam moved Kevin to big science... if he makes world-changing headways into biological problems. The outcome of this lawsuit is going to be very political. Trump involvement for sure. It's a wild ride now. Anthropic acquired Coefficient Bio for four hundred million dollars. Two ex-DeepMind computational drug discovery scientists started it. It's ten people with no revenue, yet Anthropic buys it for four hundred million to buy success.

Timelines for solving all disease are collapsing... Chan Zuckerberg Initiative originally said they wanted to cure all disease by the end of the century. The timeline is collapsing now. Eli Liy signed a massive AI drug deal with Ensilica Medicine. Boasting twenty-eight discovered drugs now. Half are in clinical trials. And half remain in proof of concept with one hundred fifteen million dollars upfront. Money is flowing now. AI developed drugs achieve an eighty-five percent Phase One success rate. Compared to the traditional fifty-two percent now. Virtual cells and FDA regulations within five years via full cell simulations using uploaded genomes. The next step is clear now. Aggiebot ships ten thousand humanoid robots globally. Moving from five to ten thousand across seventeen countries in two years. It's happening fast now. Unitry files for a six hundred million dollar IPO. Revenues up three hundred thirty-five percent year over year now. Xiaomi displays the Cyber One humanoid robot after meeting founders back in 2017 at MIT Media Lab. Founders met back in 2017. The first professional robotics league match is scheduled for April nineteenth in Boston. Racing robots fifty meters to demonstrate the singularity.

Now, the event arrives during a marathon weekend. It signals a future where automation permeates life. Robots raced fifty meters in the Seaport today. This marks a significant step for the industry. Chinese robots currently lead this charge globally. The speaker hopes US industry follows suit to distribute models to civilians rather than factories. That shift could transform two thirds of the US services sector.

Mark Cuban called this robot trend a passing phase... He claimed it is unlikely to last ten years. Yet nuance exists beyond his dismissal of the technology. Robots will become so essential that environments adapt to them. They might integrate like appliances blending seamlessly into buildings... without standing out as tools. One host joked about kicking a robot blocking their path to the bathroom.

Meanwhile US senators move to restrict Chinese robots via a bipartisan bill. They cite data theft risks similar to existing bans on Huawei chips. While China overwhelms the world market with raw physical capabilities, US citizens await models like the X Neo. The United States instead produces strong vision language action foundation models for AI integration. Gemini robotic models now integrate into twenty thousand deployed industrial robots globally.

Supply chains remain a critical bottleneck for entrepreneurs building data centers in Abilene, Texas today. They must melt metal to create electronic components because the supply chain is miles behind. This manufacturing lag contrasts sharply with virtual infrastructure moving quickly through software protocols. Investment banks find it difficult to process large IPOs for hardware companies like the six hundred ten million dollar offering.

Attention now shifts to quantum computing and Bitcoin risks facing the market today. Google moved its deadline for breaking RSA encryption from two thousand thirty five to 2029. This has caused panic among holders fearing the security implications for digital assets globally. Coinbase CEO Brian Armstrong launched a one hundred fifty million dollar coalition to roll out BIP 360. Mike Sailor argues these risks are overblown and has purchased eighty eight thousand Bitcoin valued at seven point two five billion dollars.

Market sentiment remains mixed as Bitcoin sits at seventy three thousand dollars up four thousand in five days. Banks are pulling out of the space while AI continues to suck money from other markets including cryptocurrency. The resilience of Bitcoin depends on whether protocol consensus can evolve faster than the quantum threat emerges from research labs. Momentum behind systems like the Lightning Network suggests significant underground activity remains unseen by regulators.

The conversation shifts from Bitcoin to broader implications of artificial intelligence and its economic impact on society. Some participants hold crypto via Micro Strategy others argue quantum computing is not an immediate existential threat because encryption standards can be increased easily. Instead the real risk lies in AI itself specifically clever inversion attacks against core hash functions or simply irrelevance. As AI agents emerge they may reinvent the entire crypto stack rather than adopt Bitcoin creating their own layer-one currencies.

Alex notes he holds index funds and startups viewing gold and crypto as unproductive assets compared to the efficiency of super intelligence. He argues that in a fluid economy with thousands of times more happening alignment is higher but the store of wealth could be milliseconds. Peter Diamandis counters that land could become post-scarce through AI transforming real estate into a viable asset class for investors. He suggests that if we are in the singularity why hold gold or Bitcoin when energy and compute are the ultimate store of possibility.

This leads to a discussion on wealth management where Alex describes his barbell distribution strategy for portfolio allocation. He bets that the market is a better allocator of assets and holds equity in startups directly. He does not hold gold or crypto because he struggles to see them as productive assets compared to real estate. Peter agrees citing a company with his financial interest using AI to grow new land and wanting to make real estate post-scarce.

Transitioning into a sponsored segment for Fountain Life the focus turns to heart disease prevention and early detection strategies. Dr. Don Musalem highlights that 50% of people die of heart attacks with no warning signs calling it a silent killer. Through CT angography with AI analytics they are finding that 88% of people coming in have detectable coronary artery disease. More alarming is that 23% of those individuals had soft plaque which would not traditionally be seen on CT looking at calcium scores alone.

Finally the episode moves to evidence of increasing abundance in what is called Abundance Corner for investors. Peter promotes his upcoming book We Are as Gods releasing April 14th and a program at MIT with Ray Kurzweil. They are selling tickets to those who bought 100 copies of the book with a waitlist forming quickly. Beyond that Germany just built the world's tallest windmill at 364 meters high taller than the Eiffel Tower. It is a 33 gigawatt-hour per year generation unit built inside of an old coal power plant in the Lusatia coal site.

The narrative begins with energy infrastructure transforming old industrial sites specifically a turbine being built within the Lusatia coal site in Brandenburg. This signals wind and solar penetrating old energy economics crucially acoustic sensing technology is preventing major failures in these turbines with ninety-nine percent accuracy. Identifying damage before repairs are required drops maintenance costs radically reaching abundance in energy through thousand cuts of innovation. Storage technology evolves with iron air batteries entering commercial use costing one-tenth of lithium-ion alternatives for grid storage.

Simultaneously medical breakthroughs emerge from a twelve-patient trial of redesigned CD4 immunotherapy showing extraordinary results for patients globally. Cancer vanished or entered complete remission after a single injection while six patients saw tumor shrinkage immediately following treatment. The implications suggest we are moving toward solving cancer without medical nanobots instead of retraining our body's immune systems. Water scarcity faces a different solution according to the World Bank Sub-Saharan Africa requires redistribution rather than more production.

AI technology helps optimize usage where too much water is currently wasted potentially providing all required subsistence if balanced correctly. Food production follows suit with vertical farming projected to reach forty billion dollars by 2030 currently hitting eight billion this year. Yields are three hundred fifty times greater per square foot than traditional farming utilizing ninety-five percent less water. Holland exemplifies this potential as the second-largest food exporter globally despite its small landmass proving hydroponics and aeroponics can transform logistics.

Finally education transforms through AI tutors a Wharton study found five months of coding equates to six to nine months of traditional schooling. Free and ubiquitous access allows personalized learning tailored to individual abilities understanding what students know versus do not know effectively. Critics argue forcing children through rigid lectures is cruel but AI offers compassion and adaptability that human teachers often cannot match. The future promises optimized outcomes across energy health water food and learning sectors globally.

A common cliche is Baumol's cost disease. But nuance is missing for self-motivated students. Motivated learners converse with Frontier AI models to learn faster... Yet unmotivated students need an embodiment to hold their attention. Gaming is tuned perfectly without being too difficult... It's puzzling why designers don't apply that tuning to teaching quantum physics. Players memorize weapons instantly... One could learn an entire discipline with that same brain power. The technology exists to make topics fun, yet someone needs to deliver it... What would be transformative is empowering students to do amazing things in the real world.

Shift focus... look at this exponential growth curve regarding electric vehicles sold globally. Ten thousand existed in 2010... Today we're up to twelve point seven million sold globally. The IEA predicted no million sales before 2040... Yet that same year, we sold more than one. Governments relied on these predictions and were wrong before they even spoke. The curve is still accelerating... by 2030, the data will look modest. Impact is clear: if oil prices shoot up, you're protected from volatility as we transition to solar.

Final item... The Moonshot Mates segment. We're making our debut with a rhythm in the air... it's more than just a feeling. There is a pulse, a heartbeat inside higher than the sky. Shout out to the creator community... send us your outro songs if you wish. We enjoyed back-to-back episodes with Sel Ismael, David Blondon, and AWG as my Moonshot Mates. Please subscribe to get about two episodes a week... turn on notifications so you receive it when fresh. Stay optimistic, stay hopeful; the future is ours to create. If AI happens to you... fear will be your only venture into the future. This is the most extraordinary time ever to be alive... I consider everyone a Moonshot Mate. Every week, we spend energy delivering news that matters... I invite you to join my weekly newsletter called MetaTrends. Put this into a two-minute read every week at diamandis.com/metatrens. Thank you again for joining us today... Live long and prosper, peace, and long life.""")


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

    Simple chunking — no overlap conditioning. With 2500-char chunks there are
    only 5-6 boundaries, so overlap adds complexity for minimal benefit.
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
    print("  Voxtral-4B-TTS v5 — Less Processing, More Natural (FAST Mode / 6-bit)")
    print("  Fixes: natural pauses, voice consistency, clean stitching")
    print("=" * 70)

    # Build chunks at sentence boundaries
    chunks = build_chunks(TEXT)
    total_chars = sum(len(c["text"]) for c in chunks)

    print(f"\n  Model:          {MODEL_ID}")
    print(f"  Mode:           FAST (6-bit, ~2-3x faster than bf16)")
    print(f"  Voice:          {VOICE}")
    print(f"  Temperature:    {TEMPERATURE} (lower = more consistent voice)")
    print(f"  Top-k / Top-p:  {TOP_K} / {TOP_P}")
    print(f"  Text:           {len(TEXT):,} characters")
    print(f"  Chunks:         {len(chunks)} (max {MAX_CHUNK_CHARS} chars, sentence boundaries)")
    print(f"  Stitching:      Silence gaps ({SENTENCE_GAP_MS}ms sentence / {PARAGRAPH_GAP_MS}ms paragraph)")
    print(f"  Noise reduce:   {'ON (strength={})'.format(NOISE_REDUCE_STRENGTH) if ENABLE_NOISE_REDUCTION else 'OFF'}")
    print(f"  Normalization:  Final global RMS only (preserves natural dynamics)")
    print(f"  Output:         {OUTPUT_FILE}")
    print()

    # Load model
    print("Loading model (6-bit = ~3.5GB, first run downloads from HuggingFace)...")
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
    print(f"  Generation Complete! (v5 FAST — Less Processing, More Natural)")
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
    print(f"Compare with 'output1.wav' (bf16) to check quality difference.")


if __name__ == "__main__":
    main()
