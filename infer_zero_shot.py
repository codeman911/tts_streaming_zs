"""Simple, self-contained zero-shot voice-cloning inference script for Orpheus-TTS.

Key points
-----------
1. **Prompt template identical to training**  (matches `process_item_zero_shot`):

   start_of_human  ref_text  end_of_text end_of_human
   start_of_ai  start_of_speech  ref_audio  end_of_speech end_of_ai
   start_of_human  target_text  end_of_text end_of_human
   start_of_ai  start_of_speech  ← generation begins here

2. Minimal external dependencies; no extra abstraction.
3. Portable: CPU/GPU auto-detected.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import wave
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from snac import SNAC

from orpheus_tts import OrpheusModel, tokens_decoder_sync

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Special-token constants  (must mirror training config)
# -----------------------------------------------------------------------------
TOKENS = {
    "start_of_human": 128259,
    "end_of_human": 128260,
    "start_of_ai": 128261,
    "start_of_speech": 128257,
    "end_of_speech": 128258,
    "end_of_ai": 128262,
    "end_of_text": 128009,
    "audio_tokens_start": 128266,
}

AUDIO_FRAME_SIZE = 7  # SNAC 24 kHz, 7 codes per 40 ms frame

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_audio(path: str | Path, target_sr: int = 24_000) -> torch.Tensor:
    """Load audio file as mono tensor @ target sample-rate."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform


def _deduplicate_frames(codes: List[int]) -> List[int]:
    """Remove consecutive duplicate 7-code frames (as in training)."""
    if not codes or len(codes) < AUDIO_FRAME_SIZE:
        return []
    if len(codes) % AUDIO_FRAME_SIZE:
        # Trim to full frames
        codes = codes[: -(len(codes) % AUDIO_FRAME_SIZE)]
    kept: List[int] = codes[:AUDIO_FRAME_SIZE]
    for i in range(AUDIO_FRAME_SIZE, len(codes), AUDIO_FRAME_SIZE):
        if codes[i] != kept[-AUDIO_FRAME_SIZE]:
            kept.extend(codes[i : i + AUDIO_FRAME_SIZE])
    return kept

# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------

class ZeroShotTTSInference:
    """Lightweight wrapper around Orpheus-TTS for zero-shot cloning."""

    def __init__(self, model_path: str, device: str | None = None):
        # Auto-detect device if not provided
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                # Apple-Silicon Metal backend (PyTorch ≥1.13)
                self.device = "mps"
            else:
                self.device = "cpu"
        logger.info("Using device: %s", self.device)

        # Load TTS model (vLLM currently requires GPU; on CPU/MPS we warn but still attempt)
        try:
            self.model = OrpheusModel(model_name=model_path)
        except Exception as exc:
            if self.device != "cuda":
                logger.warning(
                    "OrpheusModel failed to load on %s. GPU is recommended for vLLM-based models.\n"
                    "Error: %s", self.device, exc
                )
                raise
            raise

        # SNAC for audio tokenisation
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        if self.device in ("cuda", "mps"):
            try:
                self.snac = self.snac.to(self.device)
            except Exception:
                logger.warning("Could not move SNAC to %s; falling back to CPU", self.device)

    # ------------------------------------------------------------------
    # Audio tokenisation
    # ------------------------------------------------------------------

    def _tokenise_audio(self, waveform: torch.Tensor) -> List[int]:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # (1,1,T)
        waveform = waveform.to(dtype=torch.float32)
        if self.device in ("cuda", "mps"):
            waveform = waveform.to(self.device)

        with torch.no_grad():
            codes = self.snac.encode(waveform)["codes"]  # tuple of 3 levels
        # Interleave 7-code frames exactly like training code
        all_codes: List[int] = []
        base = TOKENS["audio_tokens_start"]
        for i in range(codes[0].shape[-1]):
            all_codes.append(codes[0][0, i].item() + base)  # q-1
            all_codes.append(codes[2][0, 4 * i].item() + base + 4096)
            all_codes.append(codes[2][0, 4 * i + 1].item() + base + 8192)
            all_codes.append(codes[1][0, 2 * i + 1].item() + base + 12288)
            all_codes.append(codes[2][0, 4 * i + 2].item() + base + 16384)
            all_codes.append(codes[2][0, 4 * i + 3].item() + base + 20480)
            all_codes.append(codes[1][0, 2 * i].item() + base + 24576)
        return _deduplicate_frames(all_codes)

    # ------------------------------------------------------------------
    # Prompt builder (matches training)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ref_text: str,
        ref_audio_codes: List[int],
        target_text: str,
    ) -> str:
        tok = self.model.tokeniser
        ref_text_ids = tok(ref_text, return_tensors="pt").input_ids[0]
        tgt_text_ids = tok(target_text, return_tensors="pt").input_ids[0]

        # tensors for special tokens
        sohu = torch.tensor([TOKENS["start_of_human"]], dtype=torch.int64)
        eohu = torch.tensor([TOKENS["end_of_human"]], dtype=torch.int64)
        soai = torch.tensor([TOKENS["start_of_ai"]], dtype=torch.int64)
        sos = torch.tensor([TOKENS["start_of_speech"]], dtype=torch.int64)
        eospeech = torch.tensor([TOKENS["end_of_speech"]], dtype=torch.int64)
        eoai = torch.tensor([TOKENS["end_of_ai"]], dtype=torch.int64)
        eot = torch.tensor([TOKENS["end_of_text"]], dtype=torch.int64)
        ref_audio_ids = torch.tensor(ref_audio_codes, dtype=torch.int64)

        prompt_ids = torch.cat(
            [
                sohu,
                ref_text_ids,
                eot,
                eohu,
                soai,
                sos,
                ref_audio_ids,
                eospeech,
                eoai,
                sohu,
                tgt_text_ids,
                eot,
                eohu,
                soai,
                sos,  # <-- generation starts after this
            ],
            dim=0,
        )
        prompt = tok.decode(prompt_ids, skip_special_tokens=False)
        # remove <s> if tokenizer adds it
        return prompt.replace("<s>", "")

    # ------------------------------------------------------------------
    # Public generate API
    # ------------------------------------------------------------------

    def generate(
        self,
        reference_audio: str,
        reference_text: str,
        target_text: str,
        output_path: str = "output.wav",
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
    ) -> str:
        # Build prompt
        waveform = _load_audio(reference_audio)
        audio_codes = self._tokenise_audio(waveform)
        prompt = self._build_prompt(reference_text, audio_codes, target_text)

        logger.info("Generating speech…")
        start = time.monotonic()
        # `generate_speech` handles decoding + waveform writing via streaming
        self.model.generate_speech(
            prompt,
            output_path,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        dur = time.monotonic() - start
        logger.info("Saved %s (%.2f s)", output_path, dur)
        return output_path

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Zero-shot TTS inference (training-aligned prompt)")
    p.add_argument("--model", required=True, help="Path to Orpheus checkpoint")
    p.add_argument("--reference_audio", required=True, help="Reference WAV/FLAC/OGG")
    p.add_argument("--reference_text", default="وخالي المعطية إلى الليلة خوي. وكان هناك فتاة المجلس الصيفي لا مش نال. ", help="Transcript of reference audio")
    p.add_argument("--target_text", required=True, help="Text to synthesise")
    p.add_argument("--output", default="output.wav", help="Destination WAV file")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    return p.parse_args()


def main():
    args = _parse_args()
    assert os.path.exists(args.reference_audio), "Reference audio not found"

    engine = ZeroShotTTSInference(args.model)
    engine.generate(
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        target_text=args.target_text,
        output_path=args.output,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()

#    وخالي المعطية إلى الليلة خوي. وكان هناك فتاة المجلس الصيفي لا مش نال.  