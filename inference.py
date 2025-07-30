#!/usr/bin/env python3
"""
Corrected Orpheus TTS Voice Cloning Implementation
Based on deep analysis of Orpheus architecture and proper token handling
"""

import os
import torch
import torchaudio
import argparse
import yaml
import logging
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import torchaudio.transforms as T

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model variables
snac_model = None
tokenizer = None
model = None
config = None

# Set deterministic seeds for reproducible results
def set_deterministic_seeds(seed=42):
    """Set deterministic seeds for reproducible generation"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this at startup
set_deterministic_seeds(42)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_models(model_path, snac_model_path):
    """Load TTS and SNAC models with proper token ID mapping"""
    global model, tokenizer, snac_model, config
    
    logger.info(f"Loading tokenizer from {model_path}")
    
    # Use fine-tuned model for better voice cloning
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=True,
        legacy=True
    )
    
    # Log special tokens for verification
    logger.info("=== SPECIAL TOKEN VERIFICATION ===")
    special_tokens = tokenizer.added_tokens_encoder
    logger.info(f"Special tokens: {special_tokens}")
    
    # Get token IDs safely using convert_tokens_to_ids with fallbacks
    token_map = {}
    
    # Define the token names we need
    token_names = {
        'start_of_human': '<|start_of_human|>',
        'end_of_human': '<|end_of_human|>',
        'start_of_ai': '<|start_of_ai|>',
        'end_of_ai': '<|end_of_ai|>',
        'start_of_speech': '<|start_of_speech|>',
        'end_of_speech': '<|end_of_speech|>',
        'end_of_text': '<|end_of_text|>',
        'start_of_text': '<|start_of_text|>'
    }
    
    # Get IDs with proper fallbacks
    for key, token in token_names.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            # Fallback to special tokens encoder
            token_id = special_tokens.get(token, None)
        if token_id is None:
            # Hardcoded fallbacks based on known Orpheus token IDs
            fallbacks = {
                'start_of_human': 128259,
                'end_of_human': 128260,
                'start_of_ai': 128261,
                'end_of_ai': 128262,
                'start_of_speech': 128257,
                'end_of_speech': 128258,
                'end_of_text': 128001,
                'start_of_text': 128256
            }
            token_id = fallbacks.get(key, 128000)  # Default fallback
        
        token_map[key] = int(token_id)
        logger.info(f"{key}: {token} -> {token_id}")
    
    logger.info(f"Final token mapping: {token_map}")
    
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model = model.to("cuda:0")
    model.eval()
    
    logger.info(f"Loading SNAC model from {snac_model_path}")
    snac_model = SNAC.from_pretrained(snac_model_path).to("cuda:0")
    snac_model.eval()
    
    return token_map

def tokenise_audio(waveform, sample_rate=24000, audio_tokens_start=128266):
    """Convert audio waveform to SNAC tokens using exact same approach as working decoder.py"""
    global snac_model
    
    try:
        # Ensure waveform is 1D for SNAC
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        # Resample if necessary
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)
        
        # Ensure proper shape: (1, 1, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to("cuda:0")
        
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        
        # Extract the codes properly
        codes_0 = codes[0][0]  # Layer 0: 1 token per frame
        codes_1 = codes[1][0]  # Layer 1: 2 tokens per frame  
        codes_2 = codes[2][0]  # Layer 2: 4 tokens per frame
        
        # Build the 7-token sequence per frame
        all_codes = []
        num_frames = codes_0.shape[0]
        
        for i in range(num_frames):
            # Layer 0: 1 token
            all_codes.append(codes_0[i].item() + audio_tokens_start)
            
            # Layer 1: 2 tokens (positions 1, 4)
            all_codes.append(codes_1[2*i].item() + audio_tokens_start + 4096)
            all_codes.append(codes_1[2*i+1].item() + audio_tokens_start + 4*4096)
            
            # Layer 2: 4 tokens (positions 2, 3, 5, 6)
            all_codes.append(codes_2[4*i].item() + audio_tokens_start + 2*4096)
            all_codes.append(codes_2[4*i+1].item() + audio_tokens_start + 3*4096)
            all_codes.append(codes_2[4*i+2].item() + audio_tokens_start + 5*4096)
            all_codes.append(codes_2[4*i+3].item() + audio_tokens_start + 6*4096)
        
        return all_codes
        
    except Exception as e:
        logger.error(f"Error in tokenise_audio: {str(e)}")
        logger.error(f"Waveform shape: {waveform.shape if 'waveform' in locals() else 'unknown'}")
        return None

def redistribute_codes(token_ids, audio_tokens_start=128266):
    """Redistribute token IDs back to SNAC layers for decoding"""
    tokens = np.array(token_ids) - audio_tokens_start
    
    # Calculate frame count
    num_tokens = len(tokens)
    if num_tokens % 7 != 0:
        tokens = tokens[:-(num_tokens % 7)]
    
    num_frames = len(tokens) // 7
    
    # Layer 0: every 7th token starting at 0
    layer_0 = tokens[0::7][:num_frames]
    
    # Layer 1: positions 1 and 4 in each 7-token frame
    layer_1 = np.concatenate([
        tokens[1::7][:num_frames],
        tokens[4::7][:num_frames]
    ])[:num_frames*2]
    
    # Layer 2: positions 2, 3, 5, 6 in each 7-token frame
    layer_2 = np.concatenate([
        tokens[2::7][:num_frames],
        tokens[3::7][:num_frames],
        tokens[5::7][:num_frames],
        tokens[6::7][:num_frames]
    ])[:num_frames*4]
    
    return [
        torch.tensor(layer_0, dtype=torch.int32).unsqueeze(0),
        torch.tensor(layer_1, dtype=torch.int32).unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.int32).unsqueeze(0)
    ]

def decode_audio_tokens(audio_tokens, audio_tokens_start=128266):
    """Decode SNAC tokens back to audio waveform using exact same approach as decoder.py"""
    global snac_model
    
    try:
        if not audio_tokens:
            logger.error("No audio tokens provided")
            return None
        
        # Convert token IDs back to SNAC codes and validate range
        tokens = np.array(audio_tokens) - audio_tokens_start
        
        # Filter out invalid tokens (must be 0-4095 for SNAC)
        valid_mask = (tokens >= 0) & (tokens < 4096)
        tokens = tokens[valid_mask]
        
        if len(tokens) == 0:
            logger.error("No valid audio tokens after filtering")
            return None
        
        logger.info(f"Filtered to {len(tokens)} valid tokens from {len(audio_tokens)} total")
        
        # Ensure we have complete frames (7 tokens per frame)
        num_tokens = len(tokens)
        if num_tokens % 7 != 0:
            tokens = tokens[:-(num_tokens % 7)]
        
        if len(tokens) == 0:
            logger.error("No complete audio frames to decode")
            return None
        
        num_frames = len(tokens) // 7
        
        # Build codes arrays properly for SNAC
        codes_0 = []
        codes_1 = []
        codes_2 = []
        
        for i in range(num_frames):
            frame_start = i * 7
            
            # Layer 0: position 0
            codes_0.append(int(tokens[frame_start]))
            
            # Layer 1: positions 1, 4
            codes_1.append(int(tokens[frame_start + 1]))
            codes_1.append(int(tokens[frame_start + 4]))
            
            # Layer 2: positions 2, 3, 5, 6
            codes_2.append(int(tokens[frame_start + 2]))
            codes_2.append(int(tokens[frame_start + 3]))
            codes_2.append(int(tokens[frame_start + 5]))
            codes_2.append(int(tokens[frame_start + 6]))
        
        # Convert to tensors with correct shape for SNAC
        # SNAC expects: [batch_size, sequence_length] for each layer
        codes = [
            torch.tensor(codes_0, dtype=torch.long).unsqueeze(0).to("cuda:0"),  # [1, num_frames]
            torch.tensor(codes_1, dtype=torch.long).unsqueeze(0).to("cuda:0"),  # [1, num_frames*2]
            torch.tensor(codes_2, dtype=torch.long).unsqueeze(0).to("cuda:0")   # [1, num_frames*4]
        ]
        
        logger.info(f"SNAC codes shapes: {[c.shape for c in codes]}")
        logger.info(f"SNAC codes ranges: {[f'{c.min().item()}-{c.max().item()}' for c in codes]}")
        
        # Validate token ranges
        for i, code in enumerate(codes):
            if torch.any(code < 0) or torch.any(code > 4096):
                logger.error(f"Invalid token values in layer {i}: {code.min().item()}-{code.max().item()}")
                return None
        
        # Decode using SNAC
        with torch.inference_mode():
            audio_hat = snac_model.decode(codes)
        
        logger.info(f"Decoded audio shape: {audio_hat.shape}")
        
        # Extract audio slice (full decoded audio)
        if len(audio_hat.shape) == 3:
            audio_slice = audio_hat[:, :, :]  # Use full decoded audio
        else:
            logger.error(f"Unexpected audio shape: {audio_hat.shape}")
            return None
            
        detached_audio = audio_slice.detach().cpu()
        
        # Convert to proper format
        audio_np = detached_audio.numpy()
        
        # Ensure we have the right shape for saving
        if len(audio_np.shape) == 3:
            audio_np = audio_np.squeeze(0)  # Remove batch dimension
        
        # Convert to int16 for saving
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        
        # Convert back to torch tensor for torchaudio
        waveform = torch.from_numpy(audio_int16.astype(np.float32) / 32767.0)
        
        # Ensure proper shape for torchaudio: [channels, samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # Remove batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        logger.info(f"Final waveform shape: {waveform.shape}")
        logger.info(f"Audio duration: {waveform.shape[-1] / 24000:.2f} seconds")
        return waveform
        
    except Exception as e:
        logger.error(f"Error decoding audio tokens: {str(e)}")
        logger.error(f"Audio tokens count: {len(audio_tokens) if audio_tokens else 0}")
        logger.error(f"Token range: {min(audio_tokens) if audio_tokens else 'None'} - {max(audio_tokens) if audio_tokens else 'None'}")
        return None

def build_reference_prompt(reference_text, reference_codes, token_map):
    """Build the reference prompt structure for voice conditioning"""
    
    # Encode reference text
    reference_text_ids = tokenizer.encode(reference_text, add_special_tokens=False)
    
    # Build reference structure
    prompt_parts = [
        [token_map['start_of_human']],      # [SOH]
        [token_map['start_of_text']],       # [SOT]
        reference_text_ids,                 # Reference text
        [token_map['end_of_text']],         # [EOT]
        [token_map['end_of_human']],        # [EOH]
        [token_map['start_of_ai']],         # [SAI]
        [token_map['start_of_speech']],     # [SOS]
        reference_codes,                    # Reference audio tokens
        [token_map['end_of_speech']],       # [EOS]
        [token_map['end_of_ai']]            # [EOAI]
    ]
    
    # Flatten the prompt
    input_ids = []
    for part in prompt_parts:
        input_ids.extend(part)
    
    return input_ids

def build_target_prompt(target_text, token_map):
    """Build the target prompt structure"""
    
    # Encode target text
    target_text_ids = tokenizer.encode(target_text, add_special_tokens=False)
    
    # Build target structure
    prompt_parts = [
        [token_map['start_of_human']],      # [SOH]
        [token_map['start_of_text']],       # [SOT]
        target_text_ids,                    # Target text
        [token_map['end_of_text']],         # [EOT]
        [token_map['end_of_human']],        # [EOH]
        [token_map['start_of_ai']],         # [SAI]
        [token_map['start_of_speech']]      # [SOS] - model generates from here
    ]
    
    # Flatten the prompt
    input_ids = []
    for part in prompt_parts:
        input_ids.extend(part)
    
    return input_ids

def generate_speech(reference_audio_path, reference_text, target_text, output_path, max_new_tokens=2000, token_map=None):
    """Generate speech using corrected zero-shot voice cloning"""
    global model, tokenizer, snac_model
    
    if token_map is None:
        logger.error("Token map not provided")
        return False
    
    # Load and preprocess reference audio
    logger.info(f"Loading reference audio from {reference_audio_path}")
    try:
        waveform, sample_rate = torchaudio.load(reference_audio_path)
        waveform = waveform.squeeze(0)
    except Exception as e:
        logger.error(f"Error loading reference audio: {e}")
        return False
    
    # Tokenize reference audio
    logger.info("Tokenizing reference audio")
    audio_tokens_start = 128266  # Fixed audio token offset
    reference_codes = tokenise_audio(waveform, sample_rate, audio_tokens_start)
    
    if reference_codes is None or len(reference_codes) == 0:
        logger.error("Failed to tokenize reference audio")
        return False
    
    logger.info(f"Generated {len(reference_codes)} reference audio tokens")
    
    # Verify all token IDs are valid integers
    for key, value in token_map.items():
        if not isinstance(value, int):
            logger.error(f"Invalid token ID for {key}: {value}")
            return False
    
    # Build complete prompt
    reference_prompt = build_reference_prompt(reference_text, reference_codes, token_map)
    target_prompt = build_target_prompt(target_text, token_map)
    
    full_prompt = reference_prompt + target_prompt
    
    # Ensure all elements are integers
    try:
        full_prompt = [int(x) for x in full_prompt]
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid token in prompt: {e}")
        logger.error(f"Prompt contains: {full_prompt}")
        return False
    
    # Convert to tensor
    input_ids = torch.tensor([full_prompt], dtype=torch.long).to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    
    # Log prompt details
    logger.info(f"Full prompt length: {len(full_prompt)} tokens")
    logger.info(f"Reference prompt length: {len(reference_prompt)} tokens")
    logger.info(f"Target prompt length: {len(target_prompt)} tokens")
    
    # Decode and log prompt
    try:
        decoded_prompt = tokenizer.decode(input_ids[0])
        logger.info(f"Decoded prompt: {decoded_prompt[:500]}...")
    except Exception as e:
        logger.error(f"Error decoding prompt: {e}")
    
    # Generate speech with optimized parameters
    logger.info("Generating speech with voice conditioning...")
    with torch.inference_mode():
        generated_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=token_map['end_of_speech']
        )
    
    # Extract only the new tokens
    new_tokens = generated_tokens[0][len(full_prompt):]
    logger.info(f"Generated {len(new_tokens)} new tokens")
    
    # Find audio tokens in generated output
    audio_tokens = []
    
    for token in new_tokens:
        if token == token_map['end_of_speech']:
            break
        if token >= audio_tokens_start:
            audio_tokens.append(token.item())
    
    if not audio_tokens:
        logger.error("No audio tokens generated")
        logger.error(f"Generated tokens: {new_tokens.tolist()}")
        return False
    
    logger.info(f"Found {len(audio_tokens)} audio tokens")
    
    # Decode audio tokens to waveform
    logger.info("Decoding audio tokens to waveform")
    output_waveform = decode_audio_tokens(audio_tokens, audio_tokens_start)
    
    if output_waveform is None:
        logger.error("Failed to decode audio tokens")
        return False
    
    # Save output audio
    logger.info(f"Saving output audio to {output_path}")
    try:
        torchaudio.save(
            output_path,
            output_waveform,
            24000,
            encoding='PCM_S',
            bits_per_sample=16
        )
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Corrected Orpheus TTS Voice Cloning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="canopylabs/orpheus-3b-0.1-ft", help="Path to model")
    # parser.add_argument("--reference-audio", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--reference-audio", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--reference-text", type=str, default=""""A lot of men, especially Arabs, get really challenged because they see what they've been told they can't have or they shouldn't have. It's wrong. They end up with women that their moms will never agree to waste three, four years of their life and hers and then get lost and then maybe go back to what they were.""", help="Reference text matching the audio")
    parser.add_argument("--output", type=str, default="output_cloned.wav", help="Path to save output audio")
    parser.add_argument("--target-text", type=str, default="""Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we’re all connected 24/7 but somehow people feel more alone than ever. <laugh> And don’t even get me started on how it’s messing with kids’ self-esteem  and mental health and whatnot.""", help="Text to synthesize in the reference voice")
    parser.add_argument("--max-new-tokens", type=int, default=2000, help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Get model paths
    model_path = args.model
    snac_model_path = config.get("snac_model", "hubertsiuzdak/snac_24khz")
    
    # Load models once and get token mapping
    token_map = load_models(model_path, snac_model_path)
    
    # Generate speech
    success = generate_speech(
        args.reference_audio,
        args.reference_text,
        args.target_text,
        args.output,
        args.max_new_tokens,
        token_map  # Pass the token_map
    )
    
    if success:
        logger.info("Voice cloning completed successfully!")
    else:
        logger.error("Voice cloning failed!")

if __name__ == "__main__":
    main()

