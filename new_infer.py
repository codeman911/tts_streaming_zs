#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Zero-Shot TTS Inference
Based on Orpheus reference implementation with optimizations
"""

import os
import torch
import torchaudio
import argparse
import yaml
import logging
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import torchaudio.transforms as T
import librosa

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

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_models(model_path, snac_model_path):
    """Load TTS and SNAC models"""
    global model, tokenizer, snac_model
    
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=8192,
        padding_side="left",
        use_fast=True,
        legacy=True
    )
    
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Move model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loading SNAC model from {snac_model_path}")
    snac_model = SNAC.from_pretrained(snac_model_path).to(device)
    snac_model.eval()

def tokenise_audio(waveform, sample_rate=24000):
    """Convert audio waveform to SNAC tokens (Orpheus style)"""
    global snac_model
    
    try:
        # Convert to torch tensor if numpy
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        waveform = waveform.to(dtype=torch.float32)
        
        # Resample if needed
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)

        # Ensure proper dimensions and move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        waveform = waveform.unsqueeze(0).to(device)

        with torch.inference_mode():
            codes = snac_model.encode(waveform)

        # Extract codes following Orpheus pattern
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))

        return all_codes
    except Exception as e:
        logger.error(f"Error in tokenise_audio: {str(e)}")
        return None

def decode_audio_tokens(audio_tokens, audio_tokens_start=128266):
    """Decode SNAC tokens back to audio waveform (EXACT copy from working inference.py)"""
    global snac_model
    
    if not audio_tokens or len(audio_tokens) < 7:
        logger.error("Not enough audio tokens to decode")
        return None
    
    # Ensure we have a multiple of 7 tokens
    if len(audio_tokens) % 7 != 0:
        audio_tokens = audio_tokens[:-(len(audio_tokens) % 7)]
    
    # Prepare arrays for each level
    level_0_tokens = []
    level_1_tokens = []
    level_2_tokens = []
    
    # Extract tokens for each level with SNAC bounds validation
    for i in range(0, len(audio_tokens), 7):
        # Level 0 token
        l0_token = audio_tokens[i] - audio_tokens_start
        l0_token = max(0, min(4095, l0_token))  # Clamp to SNAC range
        level_0_tokens.append(l0_token)
        
        # Level 1 tokens
        l1_token_1 = audio_tokens[i+1] - (audio_tokens_start + 4096)
        l1_token_1 = max(0, min(4095, l1_token_1))  # Clamp to SNAC range
        l1_token_2 = audio_tokens[i+4] - (audio_tokens_start + 4*4096)
        l1_token_2 = max(0, min(4095, l1_token_2))  # Clamp to SNAC range
        level_1_tokens.extend([l1_token_1, l1_token_2])
        
        # Level 2 tokens
        l2_token_1 = audio_tokens[i+2] - (audio_tokens_start + 2*4096)
        l2_token_1 = max(0, min(4095, l2_token_1))  # Clamp to SNAC range
        l2_token_2 = audio_tokens[i+3] - (audio_tokens_start + 3*4096)
        l2_token_2 = max(0, min(4095, l2_token_2))  # Clamp to SNAC range
        l2_token_3 = audio_tokens[i+5] - (audio_tokens_start + 5*4096)
        l2_token_3 = max(0, min(4095, l2_token_3))  # Clamp to SNAC range
        l2_token_4 = audio_tokens[i+6] - (audio_tokens_start + 6*4096)
        l2_token_4 = max(0, min(4095, l2_token_4))  # Clamp to SNAC range
        level_2_tokens.extend([l2_token_1, l2_token_2, l2_token_3, l2_token_4])
    
    # Convert to tensors
    level_0 = torch.tensor(level_0_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    level_1 = torch.tensor(level_1_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    level_2 = torch.tensor(level_2_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    
    logger.info(f"SNAC decode shapes: L0={level_0.shape}, L1={level_1.shape}, L2={level_2.shape}")
    logger.info(f"Level 0 range: {min(level_0_tokens)} - {max(level_0_tokens)}")
    logger.info(f"Level 1 range: {min(level_1_tokens)} - {max(level_1_tokens)}")
    logger.info(f"Level 2 range: {min(level_2_tokens)} - {max(level_2_tokens)}")
    
    # Decode using SNAC model
    with torch.inference_mode():
        waveform = snac_model.decode([level_0, level_1, level_2])
        # Ensure waveform is 2D [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)
    
    return waveform.cpu()

def generate_speech_orpheus_style(reference_audio_path, reference_text, target_texts, max_new_tokens=990):
    """Generate speech using Orpheus-style batching"""
    global model, tokenizer, snac_model
    
    # Load and process reference audio
    logger.info(f"Loading reference audio from {reference_audio_path}")
    audio_array, sample_rate = librosa.load(reference_audio_path, sr=24000)
    
    logger.info("Tokenizing reference audio")
    reference_codes = tokenise_audio(audio_array, sample_rate)
    if not reference_codes:
        logger.error("Failed to tokenize reference audio")
        return None
    
    # Define special tokens (Orpheus style)
    start_tokens = torch.tensor([[128259]], dtype=torch.int64)  # start_of_human
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)  # end_of_text, end_of_human, start_of_ai, start_of_speech
    final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)  # end_of_speech, end_of_ai
    
    # Tokenize reference text
    logger.info(f"Encoding reference text: {reference_text[:50]}...")
    reference_prompt_tokked = tokenizer(reference_text, return_tensors="pt")
    reference_input_ids = reference_prompt_tokked["input_ids"]
    
    # Create base prompt with reference (zeroprompt_input_ids in Orpheus)
    zeroprompt_input_ids = torch.cat([
        start_tokens,
        reference_input_ids,
        end_tokens,
        torch.tensor([reference_codes]).to(torch.int64),
        final_tokens
    ], dim=1)
    
    # Prepare batch for all target texts
    all_modified_input_ids = []
    
    # Convert single target_text to list if needed
    if isinstance(target_texts, str):
        target_texts = [target_texts]
    
    for target_text in target_texts:
        logger.info(f"Encoding target text: {target_text[:50]}...")
        target_input_ids = tokenizer(target_text, return_tensors="pt").input_ids
        
        # Create complete prompt for this target (matching working inference.py)
        second_input_ids = torch.cat([
            zeroprompt_input_ids,
            start_tokens,  # start_of_human
            target_input_ids,
            torch.tensor([[128009, 128260]], dtype=torch.int64),  # end_of_text, end_of_human
            torch.tensor([[128261]], dtype=torch.int64)  # start_of_ai (CRUCIAL!)
        ], dim=1)
        
        all_modified_input_ids.append(second_input_ids)
    
    # Batch processing with padding (Orpheus style)
    all_padded_tensors = []
    all_attention_masks = []
    
    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
    
    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        # Left pad with pad token (128263)
        padded_tensor = torch.cat([
            torch.full((1, padding), 128263, dtype=torch.int64), 
            modified_input_ids
        ], dim=1)
        # Create attention mask (0 for padding, 1 for real tokens)
        attention_mask = torch.cat([
            torch.zeros((1, padding), dtype=torch.int64), 
            torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)
        ], dim=1)
        
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)
    
    # Convert to batch tensors
    batch_input_ids = torch.cat(all_padded_tensors, dim=0).to(model.device)
    batch_attention_masks = torch.cat(all_attention_masks, dim=0).to(model.device)
    
    logger.info(f"Generating speech for {len(target_texts)} target(s)...")
    
    # Generate with Orpheus-style parameters
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,  # end_of_speech
            pad_token_id=128263
        )
    
    # Process generated sequences
    results = []
    
    for i, generated_sequence in enumerate(generated_ids):
        logger.info(f"Processing generated sequence {i+1}/{len(generated_ids)}")
        
        # Find start_of_speech token (128257) and extract audio tokens
        token_to_find = 128257  # start_of_speech
        token_to_remove = 128258  # end_of_speech
        
        # Find last occurrence of start_of_speech
        token_indices = (generated_sequence == token_to_find).nonzero(as_tuple=True)
        
        if len(token_indices[0]) > 0:
            last_occurrence_idx = token_indices[0][-1].item()
            cropped_tensor = generated_sequence[last_occurrence_idx+1:]
        else:
            logger.warning(f"No start_of_speech token found in sequence {i+1}")
            cropped_tensor = generated_sequence
        
        # Remove end_of_speech tokens
        masked_tokens = cropped_tensor[cropped_tensor != token_to_remove]
        
        # Ensure we have audio tokens and trim to multiple of 7
        if len(masked_tokens) > 0:
            row_length = masked_tokens.size(0)
            new_length = (row_length // 7) * 7
            trimmed_tokens = masked_tokens[:new_length]
            
            # Convert to audio token range (matching working inference.py)
            audio_codes = []
            for t in trimmed_tokens:
                # Convert tensor element to Python int (NO offset subtraction here)
                token_val = t.item()
                # Validate token is in reasonable range for raw tokens
                if 128266 <= token_val <= 200000:  # Raw token range
                    audio_codes.append(token_val)
                else:
                    logger.warning(f"Skipping out-of-range audio token: {token_val}")
            
            # Ensure we still have multiple of 7 after filtering
            if len(audio_codes) >= 7:
                # Trim to multiple of 7 again after filtering
                final_length = (len(audio_codes) // 7) * 7
                audio_codes = audio_codes[:final_length]
                
                logger.info(f"Decoding {len(audio_codes)} audio tokens to waveform")
                logger.info(f"Audio token range: {min(audio_codes)} - {max(audio_codes)}")
                
                waveform = decode_audio_tokens(audio_codes)
                if waveform is not None:
                    # Handle Trelis-style output (ensure proper tensor format)
                    if hasattr(waveform, 'detach'):
                        waveform = waveform.detach().squeeze().cpu()
                    
                    # Ensure waveform is 1D for torchaudio.save
                    if waveform.dim() > 1:
                        waveform = waveform.squeeze()
                    
                    results.append(waveform)
                else:
                    logger.error(f"Failed to decode audio for sequence {i+1}")
                    results.append(None)
            else:
                logger.error(f"Not enough valid audio tokens for sequence {i+1} (got {len(audio_codes)}, need at least 7)")
                results.append(None)
        else:
            logger.error(f"No audio tokens found for sequence {i+1}")
            results.append(None)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Zero-shot TTS inference (Orpheus style)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="/vast/audio/experiment/Orpheus-TTS/zer_shot_ft/checkpoints_zr/checkpoint-14720", help="Path to model checkpoint")
    parser.add_argument("--reference-audio", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--reference-text", type=str, default=""""A lot of men, especially Arabs, get really challenged because they see what they've been told they can't have or they shouldn't have. It's wrong. They end up with women that their moms will never agree to waste three, four years of their life and hers and then get lost and then maybe go back to what they were.""", help="Reference text matching the audio")
    parser.add_argument("--target-text", type=str, default="""Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we’re all connected 24/7 but somehow people feel more alone than ever. <laugh> And don’t even get me started on how it’s messing with kids’ self-esteem  and mental health and whatnot.""", help="Text to synthesize in the reference voice")
    parser.add_argument("--output", type=str, default="output_enhanced.wav", help="Path to save output audio")
    parser.add_argument("--max-new-tokens", type=int, default=990, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config(args.config)
    
    # Get model path from args or config
    model_path = args.model if args.model else config.get("model_name", args.model)
    snac_model_path = config.get("snac_model", "hubertsiuzdak/snac_24khz")
    
    # Load models
    load_models(model_path, snac_model_path)
    
    # Generate speech
    results = generate_speech_orpheus_style(
        args.reference_audio,
        args.reference_text,
        [args.target_text],  # Single target as list
        args.max_new_tokens
    )
    
    if results and results[0] is not None:
        # Save output audio
        logger.info(f"Saving output audio to {args.output}")
        waveform = results[0].unsqueeze(0)  # Add channel dimension
        
        try:
            torchaudio.save(
                args.output,
                waveform,
                24000,
                encoding='PCM_S',
                bits_per_sample=16
            )
            logger.info("Speech generation completed successfully")
            
            # Optional: Display audio info
            duration = waveform.shape[1] / 24000
            logger.info(f"Generated audio duration: {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
    else:
        logger.error("Speech generation failed")

if __name__ == "__main__":
    main()

