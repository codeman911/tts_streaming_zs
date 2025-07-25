"""Simple, self-contained zero-shot voice-cloning inference script for Orpheus-TTS.

Key points
-----------
1. **Prompt template identical to training** (matches `process_item_zero_shot`):

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
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
import torchaudio.transforms as T
from snac import SNAC

from transformers import AutoModelForCausalLM, AutoTokenizer

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
# Default batch sentences
# -----------------------------------------------------------------------------
DEFAULT_SENTENCES = [
    "Hello, how are you today?",
    "مرحبا، كيف حالك اليوم؟",
    "This is a consistency check of zero-shot cloning.",
    "هذا اختبار للتحقق من تناسق انتحال الصوت.",
    "Let's see if the voice stays the same across languages.",
]


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


# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------

class ZeroShotTTSInference:
    """Lightweight wrapper for zero-shot cloning, aligned with reference scripts."""

    def __init__(self, model_path: str, device: str | None = None):
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        logger.info("Using device: %s", self.device)

        # Load Model and Tokenizer from HuggingFace
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        # SNAC for audio tokenisation
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)
        self.model.eval()
        self.snac.eval()

    def _deduplicate_frames(self, codes_list: List[int]) -> List[int]:
        """Remove duplicate frames from a list of audio codes."""
        if not codes_list or len(codes_list) < AUDIO_FRAME_SIZE:
            return []

        if len(codes_list) % AUDIO_FRAME_SIZE != 0:
            codes_list = codes_list[:-(len(codes_list) % AUDIO_FRAME_SIZE)]

        if not codes_list:
            return []

        result = codes_list[:AUDIO_FRAME_SIZE]
        for i in range(AUDIO_FRAME_SIZE, len(codes_list), AUDIO_FRAME_SIZE):
            current_frame_start_token = codes_list[i]
            previous_frame_start_token = result[-AUDIO_FRAME_SIZE]

            if current_frame_start_token != previous_frame_start_token:
                result.extend(codes_list[i:i+AUDIO_FRAME_SIZE])

        return result

    def _tokenise_audio(self, waveform: torch.Tensor) -> List[int]:
        waveform = waveform.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            codes = self.snac.encode(waveform.unsqueeze(0))
        
        all_codes: List[int] = []
        base = TOKENS["audio_tokens_start"]
        for i in range(codes[0].shape[-1]):
            all_codes.append(codes[0][0, i].item() + base)
            all_codes.append(codes[1][0, 2 * i].item() + base + 4096)
            all_codes.append(codes[2][0, 4 * i].item() + base + 8192)
            all_codes.append(codes[2][0, 4 * i + 1].item() + base + 12288)
            all_codes.append(codes[1][0, 2 * i + 1].item() + base + 16384)
            all_codes.append(codes[2][0, 4 * i + 2].item() + base + 20480)
            all_codes.append(codes[2][0, 4 * i + 3].item() + base + 24576)
        return all_codes

    def _decode_audio(self, audio_tokens: List[int]) -> Optional[torch.Tensor]:
        if not audio_tokens or len(audio_tokens) < AUDIO_FRAME_SIZE:
            return None

        if len(audio_tokens) % AUDIO_FRAME_SIZE != 0:
            audio_tokens = audio_tokens[:-(len(audio_tokens) % AUDIO_FRAME_SIZE)]

        level_0, level_1, level_2 = [], [], []
        base = TOKENS["audio_tokens_start"]
        for i in range(0, len(audio_tokens), AUDIO_FRAME_SIZE):
            level_0.append(audio_tokens[i] - base)
            level_1.extend([audio_tokens[i+1] - (base + 4096), audio_tokens[i+4] - (base + 16384)])
            level_2.extend([audio_tokens[i+2] - (base + 8192), audio_tokens[i+3] - (base + 12288), audio_tokens[i+5] - (base + 20480), audio_tokens[i+6] - (base + 24576)])

        t_level_0 = torch.tensor(level_0, dtype=torch.long, device=self.device).unsqueeze(0)
        t_level_1 = torch.tensor(level_1, dtype=torch.long, device=self.device).unsqueeze(0)
        t_level_2 = torch.tensor(level_2, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            waveform = self.snac.decode([t_level_0, t_level_1, t_level_2])
            return waveform.squeeze(0).cpu()

    def _build_prompt_ids(self, ref_text: str, ref_audio_codes: List[int], target_text: str) -> torch.Tensor:
        ref_text_ids = self.tokenizer.encode(ref_text, add_special_tokens=False)
        tgt_text_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

        prompt_ids = (
            [TOKENS["start_of_human"]] + ref_text_ids + [TOKENS["end_of_text"], TOKENS["end_of_human"]]
            + [TOKENS["start_of_ai"], TOKENS["start_of_speech"]] + ref_audio_codes + [TOKENS["end_of_speech"], TOKENS["end_of_ai"]]
            + [TOKENS["start_of_human"]] + tgt_text_ids + [TOKENS["end_of_text"], TOKENS["end_of_human"]]
            + [TOKENS["start_of_ai"], TOKENS["start_of_speech"]]
        )
        return torch.tensor(prompt_ids, dtype=torch.long, device=self.device)

    def generate(
        self, reference_audio: str, reference_text: str, target_text: str,
        output_path: str = "output.wav", temperature: float = 0.3,
        top_p: float = 0.95, repetition_penalty: float = 1.1,
    ) -> str:
        waveform = _load_audio(reference_audio)
        audio_codes = self._tokenise_audio(waveform)
        audio_codes = self._deduplicate_frames(audio_codes)
        if not audio_codes:
            logger.error("Reference audio tokenization failed or resulted in no valid frames.")
            return ""

        prompt_ids = self._build_prompt_ids(reference_text, audio_codes, target_text)
        input_tensor = prompt_ids.unsqueeze(0)

        logger.info("Generating speech…")
        start = time.monotonic()

        # This function forces the model to start generation with the start_of_speech token.
        def prefix_allowed_tokens_fn(batch_id, sent):
            return [TOKENS["start_of_speech"]]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=4096, do_sample=True, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=TOKENS["end_of_speech"],
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        
        generated_ids = output_ids[0][len(prompt_ids):].tolist()

        audio_tokens = []
        try:
            first_audio_tok_idx = generated_ids.index(TOKENS["start_of_speech"])
            generated_ids = generated_ids[first_audio_tok_idx+1:]
        except ValueError:
            logger.warning("Could not find start_of_speech token in generated output. Using all generated tokens.")

        for token_id in generated_ids:
            if token_id == TOKENS["end_of_speech"]:
                break
            audio_tokens.append(token_id)

        if not audio_tokens:
            logger.error("No audio tokens were generated.")
            return ""

        output_waveform = self._decode_audio(audio_tokens)
        if output_waveform is None:
            logger.error("Failed to decode audio tokens into waveform.")
            return ""

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        torchaudio.save(output_path, output_waveform, 24000)

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
    p.add_argument("--reference_text", required=True, help="Transcript of reference audio")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--target_text", help="Single text to synthesise")
    group.add_argument("--sentences_file", help="Path to txt file with sentences (one per line)")
    p.add_argument("--output_folder", default="outputs", help="Directory to write WAV files")
    p.add_argument("--output", help=argparse.SUPPRESS)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    return p.parse_args()


def main():
    args = _parse_args()
    assert os.path.exists(args.reference_audio), "Reference audio not found"

    inference_engine = ZeroShotTTSInference(args.model)

    if getattr(args, "sentences_file", None):
        with open(args.sentences_file, "r", encoding="utf-8") as f:
            targets = [ln.strip() for ln in f.readlines() if ln.strip()]
    elif getattr(args, "target_text", None):
        targets = [args.target_text]
    else:
        targets = DEFAULT_SENTENCES

    os.makedirs(args.output_folder, exist_ok=True)

    for i, text in enumerate(targets):
        # Use a consistent naming scheme for output files
        output_filename = f"{i:03d}_{text[:20].replace(' ', '_')}.wav"
        if args.output and len(targets) == 1:
             # If a specific output file is requested for a single target, use it
            output_filename = args.output

        output_path = os.path.join(args.output_folder, output_filename)
        logger.info("--- Generating for: '%s' ---", text)
        inference_engine.generate(
            reference_audio=args.reference_audio,
            reference_text=args.reference_text,
            target_text=text,
            output_path=output_path,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    for idx, sentence in enumerate(targets, 1):
        safe_name = f"{idx}.wav"
        out_path = os.path.join(args.output_folder, safe_name)
        cmd = [
            sys.executable,
            script_path,
            "--model", args.model,
            "--reference_audio", args.reference_audio,
            "--reference_text", args.reference_text,
            "--target_text", sentence,
            "--output", out_path,
            "--temperature", str(args.temperature),
            "--repetition_penalty", str(args.repetition_penalty),
        ]
        logger.info("[Batch] Running: %s", shlex.join(cmd))
        subprocess.run(cmd, check=True)



if __name__ == "__main__":
    main()




