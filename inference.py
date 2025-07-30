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
    global model, tokenizer, snac_model, config
    
    logger.info(f"Loading tokenizer from {model_path}")
    # Load tokenizer without vocabulary restrictions
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=True,
        legacy=True  # Important for maintaining original tokenizer behavior
    )
    
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True  # Allow model to use its original configuration
    )
    
    # Move model to GPU
    model = model.to("cuda:0")
    model.eval()
    
    logger.info(f"Loading SNAC model from {snac_model_path}")
    snac_model = SNAC.from_pretrained(snac_model_path).to("cuda:0")
    snac_model.eval()

def tokenise_audio(waveform, sample_rate=24000, audio_tokens_start=128266):
    """Convert audio waveform to SNAC tokens"""
    global snac_model
    
    try:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        waveform = waveform.to(dtype=torch.float32)
        
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)

        # Always use cuda:0
        waveform = waveform.unsqueeze(0).to("cuda:0")

        with torch.inference_mode():
            codes = snac_model.encode(waveform)

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + audio_tokens_start)
            all_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)
            all_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + audio_tokens_start + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + audio_tokens_start + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + audio_tokens_start + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + audio_tokens_start + (6*4096))

        return all_codes
    except Exception as e:
        logger.error(f"Error in tokenise_audio: {str(e)}")
        return None

def remove_duplicate_frames(codes_list):
    """Remove duplicate frames from codes_list"""
    if codes_list is None or len(codes_list) == 0:
        return None
        
    if len(codes_list) % 7 != 0:
        # Truncate to nearest multiple of 7
        codes_list = codes_list[:-(len(codes_list) % 7)]
    
    if len(codes_list) == 0:
        return None
        
    result = codes_list[:7]
    
    for i in range(7, len(codes_list), 7):
        # Check if we have a complete frame
        if i+6 >= len(codes_list):
            break
            
        current_first = codes_list[i]
        previous_first = result[-7]
        
        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
    
    # Final check to ensure we have at least one frame
    if len(result) < 7:
        return None
        
    return result

def decode_audio_tokens(audio_tokens, audio_tokens_start=128266):
    """Decode SNAC tokens back to audio waveform"""
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
    
    # Extract tokens for each level
    for i in range(0, len(audio_tokens), 7):
        level_0_tokens.append(audio_tokens[i] - audio_tokens_start)
        level_1_tokens.extend([
            audio_tokens[i+1] - (audio_tokens_start + 4096),
            audio_tokens[i+4] - (audio_tokens_start + 4*4096)
        ])
        level_2_tokens.extend([
            audio_tokens[i+2] - (audio_tokens_start + 2*4096),
            audio_tokens[i+3] - (audio_tokens_start + 3*4096),
            audio_tokens[i+5] - (audio_tokens_start + 5*4096),
            audio_tokens[i+6] - (audio_tokens_start + 6*4096)
        ])
    
    # Convert to tensors
    level_0 = torch.tensor(level_0_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    level_1 = torch.tensor(level_1_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    level_2 = torch.tensor(level_2_tokens, dtype=torch.long).unsqueeze(0).to("cuda:0")
    
    # Decode using SNAC model
    with torch.inference_mode():
        waveform = snac_model.decode([level_0, level_1, level_2])
        # Ensure waveform is 2D [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)
    
    return waveform.cpu()

def generate_speech(reference_audio_path, reference_text, target_text, output_path, max_new_tokens=2000):
    """Generate speech using zero-shot voice cloning"""
    global model, tokenizer, snac_model, config
    
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
    reference_codes = tokenise_audio(waveform, sample_rate)
    if reference_codes is None or len(reference_codes) == 0:
        logger.error("Failed to tokenize reference audio")
        return False
    
    # Get special token IDs
    start_of_human = tokenizer.convert_tokens_to_ids('<|start_of_human|>')
    end_of_human = tokenizer.convert_tokens_to_ids('<|end_of_human|>')
    start_of_ai = tokenizer.convert_tokens_to_ids('<|start_of_ai|>')
    end_of_ai = tokenizer.convert_tokens_to_ids('<|end_of_ai|>')
    start_of_speech = tokenizer.convert_tokens_to_ids('<|start_of_speech|>')
    end_of_speech = tokenizer.convert_tokens_to_ids('<|end_of_speech|>')
    end_of_text = tokenizer.convert_tokens_to_ids('<|end_of_text|>')
    audio_tokens_start = config.get('audio_tokens_start', 128266)
    
    logger.info(f"Special token IDs: start_of_human={start_of_human}, end_of_human={end_of_human}, start_of_ai={start_of_ai}, end_of_ai={end_of_ai}")
    logger.info(f"start_of_speech={start_of_speech}, end_of_speech={end_of_speech}, end_of_text={end_of_text}")
    
    # Encode reference and target text
    logger.info("Encoding reference and target text")
    reference_text_ids = tokenizer.encode(reference_text, add_special_tokens=False)
    target_text_ids = tokenizer.encode(target_text, add_special_tokens=False)
    
    # Build prompt exactly as per training format
    prompt_parts = [
        [start_of_human],                    # Start of reference
        reference_text_ids,                  # Reference text
        [end_of_text, end_of_human],         # End reference
        [start_of_ai, start_of_speech],      # Start AI + speech
        reference_codes,                     # Reference audio tokens
        [end_of_speech, end_of_ai],          # End speech + AI
        [start_of_human],                    # Start of target
        target_text_ids,                     # Target text
        [end_of_text, end_of_human]          # End target
    ]
    
    # Flatten the prompt
    input_ids = []
    for part in prompt_parts:
        input_ids.extend(part)
    
    input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda:0")
    
    # Log prompt details
    logger.info(f"Prompt length: {len(input_ids[0])} tokens")
    logger.info(f"First 50 tokens: {input_ids[0][:50].tolist()}")
    logger.info(f"Last 50 tokens: {input_ids[0][-50:].tolist()}")
    
    # Decode and log the prompt
    try:
        decoded_prompt = tokenizer.decode(input_ids[0])
        logger.info(f"Decoded prompt: {decoded_prompt[:500]}...")
    except Exception as e:
        logger.error(f"Error decoding prompt: {e}")
    
    # Generate with deterministic settings
    logger.info("Generating speech with deterministic settings")
    with torch.inference_mode():
        generated_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic generation
            temperature=1.0,  # Neutral temperature
            top_p=1.0,        # Disable top-p sampling
            repetition_penalty=1.0,  # No repetition penalty
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            forced_bos_token_id=start_of_ai  # Force start with start_of_ai
        )
    
    # Extract only the new tokens
    new_tokens = generated_tokens[0][len(input_ids[0]):]
    logger.info(f"Generated {len(new_tokens)} new tokens")
    
    # Log first 50 generated tokens
    if len(new_tokens) > 0:
        logger.info(f"First 50 generated tokens: {new_tokens[:50].tolist()}")
        try:
            logger.info(f"Generated text: {tokenizer.decode(new_tokens[:50])}...")
        except Exception as e:
            logger.error(f"Error decoding generated tokens: {e}")
    
    # Find audio tokens in the generated output
    audio_tokens = []
    in_speech = False
    
    for token in new_tokens:
        if token == start_of_speech:
            in_speech = True
            continue
        elif token == end_of_speech:
            in_speech = False
            break
        
        if in_speech and token >= audio_tokens_start:
            audio_tokens.append(token)
    
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
    
    # Ensure output_waveform is 2D [channels, samples]
    if output_waveform.dim() != 2:
        logger.info(f"Reshaping output waveform from shape {output_waveform.shape}")
        output_waveform = output_waveform.view(1, -1)
    
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
    parser = argparse.ArgumentParser(description="Zero-shot TTS inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str,default="/vast/audio/experiment/Orpheus-TTS/zer_shot_ft/checkpoints_zr/checkpoint-14720", help="Path to model checkpoint (overrides config)")
    parser.add_argument("--reference-audio", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--reference-text", type=str, default=""""A lot of men, especially Arabs, get really challenged because they see what they've been told they can't have or they shouldn't have. It's wrong. They end up with women that their moms will never agree to waste three, four years of their life and hers and then get lost and then maybe go back to what they were.""", help="Reference text matching the audio")
    parser.add_argument("--target-text", type=str, default="""Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we’re all connected 24/7 but somehow people feel more alone than ever. <laugh> And don’t even get me started on how it’s messing with kids’ self-esteem  and mental health and whatnot.""", help="Text to synthesize in the reference voice")
    parser.add_argument("--output", type=str, default="output_1.wav", help="Path to save output audio")
    parser.add_argument("--max-new-tokens", type=int, default=10000, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config(args.config)
    
    # Get model path from args or config
    model_path = args.model if args.model else config["model_name"]
    snac_model_path = config.get("snac_model", "hubertsiuzdak/snac_24khz")
    
    # Load models
    load_models(model_path, snac_model_path)
    
    # Generate speech
    success = generate_speech(
        args.reference_audio,
        args.reference_text,
        args.target_text,
        args.output,
        args.max_new_tokens
    )
    
    if success:
        logger.info("Speech generation completed successfully")
    else:
        logger.error("Speech generation failed")

if __name__ == "__main__":
    main()