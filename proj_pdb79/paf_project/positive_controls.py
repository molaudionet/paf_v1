#!/usr/bin/env python3
"""
Positive Control Design for Paper 1
====================================
Before spending API credits on 200-400 BBBP molecules, run these
controls to confirm the LLM is actually *processing* the audio input
and not just guessing or ignoring it.

Three levels of controls, from simplest to most informative.
"""

# =============================================================================
# CONTROL 1: "Is the model listening at all?" (Discrimination control)
# =============================================================================
#
# Problem it solves:
#   If you send audio to GPT-4o-mini and it always says "YES" regardless,
#   you can't tell whether it's ignoring audio or just biased.
#
# Design:
#   Send two CLEARLY different audio signals and ask "are these the same?"
#
#   - Signal A: low-pitched continuous tone (e.g., 200 Hz sine, 3 seconds)
#   - Signal B: high-pitched staccato (e.g., 2000 Hz beeps, 3 seconds)
#
#   Prompt: "You will hear an audio clip. Is this clip a LOW continuous
#            tone or a HIGH staccato pattern? Answer LOW or HIGH."
#
#   Run 20 times each (40 total). If accuracy < 90%, the model is not
#   meaningfully processing audio → stop, don't waste API on molecules.
#
#   Cost: ~$0.50. Time: 10 minutes.

import numpy as np
import soundfile as sf
import os


def generate_control1_audio(out_dir: str, sr: int = 16000, duration: float = 3.0):
    """Generate simple discrimination control audio pairs."""
    os.makedirs(out_dir, exist_ok=True)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Signal A: low continuous tone (200 Hz)
    for i in range(20):
        sig = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
        sf.write(os.path.join(out_dir, f"ctrl1_low_{i:02d}.wav"), sig, sr)

    # Signal B: high staccato (2000 Hz, 100ms on / 100ms off)
    for i in range(20):
        envelope = np.zeros(n, dtype=np.float32)
        on_samples = int(0.1 * sr)
        off_samples = int(0.1 * sr)
        cycle = on_samples + off_samples
        for j in range(n):
            if (j % cycle) < on_samples:
                envelope[j] = 1.0
        sig = (0.3 * np.sin(2 * np.pi * 2000 * t) * envelope).astype(np.float32)
        sf.write(os.path.join(out_dir, f"ctrl1_high_{i:02d}.wav"), sig, sr)

    print(f"Control 1: wrote 40 audio files to {out_dir}")


# =============================================================================
# CONTROL 2: "Can it distinguish molecular-like audio?" (Complexity control)
# =============================================================================
#
# Problem it solves:
#   Control 1 uses trivial audio. But molecular sonification produces
#   *complex* timbral signals. Maybe the LLM processes simple tones but
#   fails on complex spectral content.
#
# Design:
#   Generate two classes of SYNTHETIC "molecular-like" audio:
#
#   Class A ("heavy molecule"): many low-pitched overlapping harmonics,
#            long sustain, dense texture → mimics high-mass, many-atom molecule
#   Class B ("light molecule"): few high-pitched tones, sparse, short
#            attacks → mimics low-mass, few-atom molecule
#
#   Prompt: "This audio represents a structured scientific encoding.
#            Based on acoustic properties (pitch, density, complexity),
#            classify this as HEAVY or LIGHT. Answer HEAVY or LIGHT."
#
#   Run 20 each (40 total). If accuracy < 75%, the model struggles with
#   complex spectral discrimination → your molecular results need careful
#   interpretation.
#
#   Cost: ~$0.50. Time: 10 minutes.

def generate_control2_audio(out_dir: str, sr: int = 16000, duration: float = 6.0):
    """Generate molecular-complexity-like control audio."""
    os.makedirs(out_dir, exist_ok=True)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(42)

    # Class A: "heavy" — low fundamentals, many harmonics, dense
    for i in range(20):
        sig = np.zeros(n, dtype=np.float64)
        base_freq = rng.uniform(80, 200)
        n_harmonics = rng.integers(8, 15)
        for h in range(1, n_harmonics + 1):
            amp = 0.15 / h
            freq = base_freq * h + rng.normal(0, 2)
            phase = rng.uniform(0, 2 * np.pi)
            sig += amp * np.sin(2 * np.pi * freq * t + phase)
        sig = (sig / (np.max(np.abs(sig)) + 1e-8) * 0.3).astype(np.float32)
        sf.write(os.path.join(out_dir, f"ctrl2_heavy_{i:02d}.wav"), sig, sr)

    # Class B: "light" — high fundamentals, few harmonics, sparse
    for i in range(20):
        sig = np.zeros(n, dtype=np.float64)
        base_freq = rng.uniform(1000, 3000)
        n_harmonics = rng.integers(2, 4)
        for h in range(1, n_harmonics + 1):
            amp = 0.2 / h
            freq = base_freq * h + rng.normal(0, 5)
            # Add envelope to make it sparse/plucky
            attack = np.exp(-t * rng.uniform(1, 4))
            sig += amp * np.sin(2 * np.pi * freq * t) * attack
        sig = (sig / (np.max(np.abs(sig)) + 1e-8) * 0.3).astype(np.float32)
        sf.write(os.path.join(out_dir, f"ctrl2_light_{i:02d}.wav"), sig, sr)

    print(f"Control 2: wrote 40 audio files to {out_dir}")


# =============================================================================
# CONTROL 3: "Null control — does it hallucinate signal from noise?"
# =============================================================================
#
# Problem it solves:
#   LLMs are notorious for confident answers even on garbage input.
#   If the model classifies random noise as BBB+ or BBB- with 60%
#   "accuracy" on a balanced set, your real result means nothing.
#
# Design:
#   Generate 50 random noise clips (same duration, SR, RMS as real audio).
#   Ask the SAME BBBP prompt. Assign random 50/50 labels.
#   Measure "accuracy" — it should be ~50% ± noise.
#
#   If "accuracy" on noise is significantly > 50% → the model is
#   hallucinating structure, and your real result is suspect.
#
#   Also measure: does the model give uniform YES/NO on noise,
#   or is it biased toward one answer? Report the bias rate.
#
#   Cost: ~$0.25. Time: 5 minutes.

def generate_control3_audio(out_dir: str, sr: int = 16000, duration: float = 6.0,
                            rms_target: float = 0.05):
    """Generate noise control audio (matched to molecular audio specs)."""
    os.makedirs(out_dir, exist_ok=True)
    n = int(sr * duration)
    rng = np.random.default_rng(99)

    for i in range(50):
        noise = rng.normal(0, 1, n).astype(np.float32)
        # Normalize to match target RMS
        current_rms = np.sqrt(np.mean(noise ** 2))
        noise = noise * (rms_target / (current_rms + 1e-8))
        sf.write(os.path.join(out_dir, f"ctrl3_noise_{i:02d}.wav"), noise, sr)

    print(f"Control 3: wrote 50 noise files to {out_dir}")


# =============================================================================
# RUN ORDER AND DECISION LOGIC
# =============================================================================
#
# Step 1: Run Control 1 (simple discrimination)
#   → If accuracy < 90%: STOP. Model isn't processing audio.
#     Diagnose: maybe wrong API endpoint, audio not transmitted, etc.
#
# Step 2: Run Control 2 (complex spectral discrimination)
#   → If accuracy < 75%: CAUTION. Model struggles with complex audio.
#     Proceed but interpret molecular results very conservatively.
#   → If accuracy >= 75%: Good. Complex audio is being processed.
#
# Step 3: Run Control 3 (noise null)
#   → If "accuracy" on noise > 60%: WARNING. Model hallucinates.
#     Check for YES/NO bias. If biased (e.g., 80% YES), you need
#     to correct for base rate in your real analysis.
#   → If ~50%: Clean null. Proceed with confidence.
#
# Step 4: Only THEN run the real BBBP experiment.
#
# Total cost for all controls: ~$1.25
# Total time: ~30 minutes
# Value: prevents wasting $20-50 on uninterpretable results
#
# =============================================================================

# =============================================================================
# REPORTING: What goes in the paper
# =============================================================================
#
# In Methods:
#   "Prior to molecular evaluation, we conducted positive and null controls
#    to verify that models meaningfully process audio input. Models achieved
#    X% accuracy on simple audio discrimination (Control 1), Y% on complex
#    spectral classification (Control 2), and Z% on random noise with
#    arbitrary labels (Control 3, expected ~50%)."
#
# In Supplementary:
#   Full control results table.
#   Audio generation parameters.
#   Bias rate on noise control.
#
# This makes your paper MUCH harder to dismiss.
# =============================================================================


if __name__ == "__main__":
    base_dir = "control_audio"
    generate_control1_audio(os.path.join(base_dir, "ctrl1_discrimination"))
    generate_control2_audio(os.path.join(base_dir, "ctrl2_complexity"))
    generate_control3_audio(os.path.join(base_dir, "ctrl3_noise_null"))

    print("\n--- ALL CONTROLS GENERATED ---")
    print("Next: send to LLM APIs with appropriate prompts (see docstrings above)")
    print("Decision tree:")
    print("  Control 1 < 90%  → STOP (model not processing audio)")
    print("  Control 2 < 75%  → CAUTION (complex audio difficult)")
    print("  Control 3 > 60%  → WARNING (model hallucinating)")
    print("  All pass         → PROCEED to real BBBP experiment")
