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
    global model, tokenizer, config, snac_model
    
    # Load reference audio and process SNAC codes as before
    logger.info(f"Loading reference audio from {reference_audio_path}")
    waveform, sample_rate = torchaudio.load(reference_audio_path)
    if waveform.shape[0] > 1:  # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Get SNAC codes for reference audio
    logger.info("Tokenizing reference audio")
    audio_tokens_start = config.get("audio_tokens_start", 128266)
    reference_codes = tokenise_audio(waveform, sample_rate, audio_tokens_start)
    if reference_codes is None or len(reference_codes) == 0:
        logger.error("Failed to tokenize reference audio")
        return False
    
    # Remove duplicate frames
    reference_codes = remove_duplicate_frames(reference_codes)
    if reference_codes is None or len(reference_codes) == 0:
        logger.error("No valid frames in reference audio")
        return False
    
    # Get vocabulary size
    vocab_size = tokenizer.vocab_size
    logger.info(f"Model vocabulary size: {vocab_size}")
    
    # Create token IDs from config
    start_of_human = config.get("start_of_human", 128259)
    end_of_human = config.get("end_of_human", 128260)
    start_of_ai = config.get("start_of_ai", 128261)
    start_of_speech = config.get("start_of_speech", 128257)
    end_of_speech = config.get("end_of_speech", 128258)
    end_of_ai = config.get("end_of_ai", 128262)
    end_of_text = config.get("end_of_text", 128009)
    
    # Encode texts directly without validation
    logger.info(f"Encoding reference text: {reference_text[:50]}...")
    reference_text_ids = tokenizer.encode(reference_text, add_special_tokens=True)
    
    logger.info(f"Encoding target text: {target_text[:50]}...")
    target_text_ids = tokenizer.encode(target_text, add_special_tokens=True)
    
    # Create input sequence without vocabulary checks
    input_ids = (
        [start_of_human] 
        + reference_text_ids 
        + [end_of_text, end_of_human]
        + [start_of_ai] 
        + [start_of_speech] 
        + reference_codes 
        + [end_of_speech] 
        + [end_of_ai]
        + [start_of_human] 
        + target_text_ids 
        + [end_of_text, end_of_human]
        + [start_of_ai]
    )
    
    # Generate with similar parameters to test_inference.py
    # Generate with correct parameters
    input_tensor = torch.tensor([input_ids], device=model.device)
    logger.info("Generating speech...")
    with torch.inference_mode():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=config.get("pad_token", 128263),
            eos_token_id=end_of_speech
        )
        logger.info(f"Raw model output shape: {output.shape}")
        logger.info(f"Raw model output first 50 tokens: {output[0][:50].tolist()}")
    
        # Extract generated tokens (only the new ones)
        generated_tokens = output[0][len(input_ids):].tolist()
        logger.info(f"Number of generated tokens: {len(generated_tokens)}")
        logger.info(f"First 50 generated tokens: {generated_tokens[:50]}")
        try:
            logger.info(f"Generated text: {tokenizer.decode(generated_tokens[:50])}...")
        except Exception as e:
            logger.error(f"Error decoding generated tokens: {e}")
    
    # Find audio tokens in the generated output
    audio_tokens = []
    in_speech = False
    
    for token in generated_tokens:
        if token == start_of_speech:
            in_speech = True
            continue
        elif token == end_of_speech:
            in_speech = False
            break
        
        if in_speech and token >= audio_tokens_start:  # Removed vocab_size check since we're using extended vocabulary
            audio_tokens.append(token)
    
    if not audio_tokens:
        logger.error("No audio tokens generated")
        return False
    
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

    parser.add_argument("--output-folder", type=str, default="output", help="Path to save output audio")
    parser.add_argument("--max-new-tokens", type=int, default=10000, help="Maximum number of new tokens to generate")
    parser.add_argument("--target-texts", type=str, nargs='+', help="List of target texts or path to .txt file containing target texts")
    
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config(args.config)
    
    # Get model path from args or config
    model_path = args.model if args.model else config["model_name"]
    snac_model_path = config.get("snac_model", "hubertsiuzdak/snac_24khz")

    # Load target texts from file if provided
    if len(args.target_texts) == 1 and args.target_texts[0].endswith('.txt'):
        with open(args.target_texts[0], 'r') as f:
            args.target_texts = [line.strip() for line in f.readlines()]
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Generate speech for each target text, reloading model/SNAC per sentence (state leakage test)
    for i, target_text in enumerate(args.target_texts):
        logger.info(f"[STATE-LEAK TEST] Reloading model and SNAC for sentence {i+1}/{len(args.target_texts)}")
        load_models(model_path, snac_model_path)
        output_path = os.path.join(args.output_folder, f"{i:03}.wav")
        success = generate_speech(
            args.reference_audio,
            args.reference_text,
            target_text,
            output_path,
            args.max_new_tokens
        )
        
        if success:
            logger.info(f"Speech generation completed successfully for {target_text[:50]}...")
        else:
            logger.error(f"Speech generation failed for {target_text[:50]}...")
    
    logger.info("All speech generations completed")

if __name__ == "__main__":
    main()
