import json
import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset, concatenate_datasets
import yaml
from snac import SNAC
import torchaudio.transforms as T
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import multiprocessing as mp
from itertools import islice
import gc
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import random

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set higher threshold for certain loggers to reduce noise
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# Global model variables
model = None
tokenizer = None
config = None

def tokenise_audio(waveform, sample_rate=24000, audio_tokens_start=128266):
    """Convert audio waveform to SNAC tokens"""
    global model
    
    try:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        waveform = waveform.to(dtype=torch.float32)
        
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)

        # Always use cuda:0 like in tokenise_speech_dataset.py
        waveform = waveform.unsqueeze(0).to("cuda:0")

        with torch.inference_mode():
            codes = model.encode(waveform)

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
        logger.warning(f"Error in tokenise_audio: {str(e)}")
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

def find_paired_utterance(item, manifest_items, speaker_utterances=None):
    """Find a paired utterance from the same speaker for zero-shot training"""
    
    # Get speaker information
    speaker = item.get("speaker", item.get("voice_name", "unknown_speaker"))
    current_audio_path = item.get("audio_filepath", "")
    
    # If we have a pre-built speaker utterances dictionary, use it
    if speaker_utterances and speaker in speaker_utterances:
        # Get all utterances from this speaker
        utterances = speaker_utterances[speaker]
        # Filter out the current utterance
        other_utterances = [u for u in utterances if u.get("audio_filepath") != current_audio_path]
        
        if other_utterances:
            # Select a random utterance from the same speaker
            return random.choice(other_utterances)
    
    # Fallback: search through manifest items
    if manifest_items:
        # Filter items by the same speaker
        same_speaker_items = [i for i in manifest_items 
                             if (i.get("speaker") == speaker or i.get("voice_name") == speaker) 
                             and i.get("audio_filepath") != current_audio_path]
        
        if same_speaker_items:
            # Select a random item from the same speaker
            return random.choice(same_speaker_items)
    
    # No paired utterance found
    return None

def save_speaker_statistics(speaker_utterances, output_dir):
    """Save speaker statistics to a JSON file"""
    speaker_stats = {}
    
    for speaker, utterances in speaker_utterances.items():
        total_duration = sum(u.get("duration", 0) for u in utterances)
        total_utterances = len(utterances)
        
        speaker_stats[speaker] = {
            "total_utterances": total_utterances,
            "total_duration_seconds": round(total_duration, 2),
            "average_duration": round(total_duration / total_utterances, 2) if total_utterances > 0 else 0
        }
    
    # Save to file
    stats_path = os.path.join(output_dir, "speaker_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(speaker_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved speaker statistics for {len(speaker_stats)} speakers to {stats_path}")
    return stats_path

def validate_token_structure(input_ids, labels, config):
    """Validate token structure to ensure proper formatting"""
    # Extract special tokens from config
    start_of_human = config.get("start_of_human", 128259)
    end_of_human = config.get("end_of_human", 128260)
    start_of_ai = config.get("start_of_ai", 128261)
    start_of_speech = config.get("start_of_speech", 128257)
    end_of_speech = config.get("end_of_speech", 128258)
    end_of_ai = config.get("end_of_ai", 128262)
    audio_tokens_start = config.get("audio_tokens_start", 128266)
    
    # Basic validation checks
    validation = {
        "is_prefix": all(a == b for a, b in zip(input_ids, labels[:len(input_ids)])),
        "starts_with_human": input_ids[0] == start_of_human,
        "has_audio_tokens": any(t >= audio_tokens_start for t in input_ids),
        "has_special_tokens": (
            start_of_ai in input_ids and 
            start_of_speech in input_ids and
            end_of_speech in input_ids and
            end_of_ai in input_ids
        )
    }
    
    return all(validation.values()), validation

def process_item_zero_shot(item, manifest_items=None, speaker_utterances=None):
    """Process a single audio-text pair for zero-shot voice cloning training"""
    global config, tokenizer
    
    try:
        # Check if file exists
        audio_path = item.get("audio_filepath")
        if not audio_path or not os.path.exists(audio_path):
            # Changed from debug to avoid excessive logs
            return None
            
        # Check duration
        min_duration = config.get("min_duration", 0.1)
        max_duration = config.get("max_duration", 40.0)
        
        # Get speaker information for adaptive duration handling
        speaker = item.get("speaker", item.get("voice_name", "unknown_speaker"))
        
        # Enforce minimum 3.0 seconds for reference audio by default
        reference_min_duration = 3.0
        item_duration = item.get("duration", 0)
        
        # Check if this speaker has any audio files >= 3.0 seconds
        has_long_audio = False
        if speaker_utterances and speaker in speaker_utterances:
            long_utterances = [u for u in speaker_utterances[speaker] if u.get("duration", 0) >= reference_min_duration]
            has_long_audio = len(long_utterances) > 0
        
        # If speaker has no long audio files, fall back to shorter minimum (but still require at least 1.0 second)
        if not has_long_audio:
            fallback_min_duration = 1.0
            if item_duration >= fallback_min_duration:
                # Only log this once per speaker to avoid spam
                if not hasattr(process_item_zero_shot, "logged_short_speakers"):
                    process_item_zero_shot.logged_short_speakers = set()
                
                if speaker not in process_item_zero_shot.logged_short_speakers:
                    logger.warning(f"Speaker '{speaker}' has no audio files >= {reference_min_duration}s. Using fallback minimum of {fallback_min_duration}s.")
                    process_item_zero_shot.logged_short_speakers.add(speaker)
                
                # Use the shorter audio file
                reference_min_duration = fallback_min_duration
        
        # Apply the appropriate minimum duration
        if not (reference_min_duration <= item_duration <= max_duration):
            return None
            
        # Load and process reference audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # Convert stereo to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Get SNAC codes for reference audio
        audio_tokens_start = config.get("audio_tokens_start", 128266)
        reference_codes = tokenise_audio(waveform, sample_rate, audio_tokens_start)
        if reference_codes is None or len(reference_codes) == 0:
            return None
            
        # Remove duplicate frames
        reference_codes = remove_duplicate_frames(reference_codes)
        if reference_codes is None or len(reference_codes) == 0:
            return None
            
        # Get reference text from the example
        reference_text = item.get("text", "")
        if not reference_text:
            return None
        
        # Find a paired utterance from the same speaker for target generation
        paired_item = find_paired_utterance(item, manifest_items, speaker_utterances)
        
        # If no paired utterance found, we can't create a zero-shot example
        if not paired_item:
            # Changed from debug to avoid excessive logs
            return None
            
        # Get target text
        target_text = paired_item.get("text", "")
        if not target_text:
            return None
            
        # Load target audio for training labels
        target_audio_path = paired_item.get("audio_filepath")
        if not target_audio_path or not os.path.exists(target_audio_path):
            return None
            
        # Load and process target audio
        target_waveform, target_sample_rate = torchaudio.load(target_audio_path)
        if target_waveform.shape[0] > 1:  # Convert stereo to mono
            target_waveform = torch.mean(target_waveform, dim=0, keepdim=True)
            
        # Get SNAC codes for target audio
        target_codes = tokenise_audio(target_waveform, target_sample_rate, audio_tokens_start)
        if target_codes is None or len(target_codes) == 0:
            return None
            
        # Remove duplicate frames
        target_codes = remove_duplicate_frames(target_codes)
        if target_codes is None or len(target_codes) == 0:
            return None
        
        # Create token IDs from config - using the same format as trelis_orpheus_clone.py
        start_of_human = config.get("start_of_human", 128259)
        end_of_human = config.get("end_of_human", 128260)
        start_of_ai = config.get("start_of_ai", 128261)
        start_of_speech = config.get("start_of_speech", 128257)
        end_of_speech = config.get("end_of_speech", 128258)
        end_of_ai = config.get("end_of_ai", 128262)
        end_of_text = config.get("end_of_text", 128009)
        
        # Encode reference text
        reference_text_ids = tokenizer.encode(reference_text, add_special_tokens=True)
        
        # Encode target text
        target_text_ids = tokenizer.encode(target_text, add_special_tokens=True)
        
        # Create zero-shot voice cloning format exactly like in trelis_orpheus_clone.py
        # Format: start_of_human, text, end_of_text, end_of_human, start_of_ai, start_of_speech, 
        #         speech, end_of_speech, end_of_ai, start_of_human, text, end_of_human
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
        )
        
        # For training labels, we need the expected output which includes the generated speech
        # This follows the same pattern but includes the AI response with target audio
        labels = (
            input_ids
            + [start_of_ai]
            + [start_of_speech]
            + target_codes
            + [end_of_speech]
            + [end_of_ai]
        )
        
        # Add detailed logging for debugging (log every 1000th item instead of every 100th)
        speaker = item.get("speaker", item.get("voice_name", "unknown_speaker"))
        item_id = hash(audio_path) % 10000  # Create a simple hash for identification
        
        if item_id % 1000 == 0:  # Reduced logging frequency from 100 to 1000
            # Simplified logging output
            logger.info(f"Processing example: Speaker={speaker}, RefAudio={os.path.basename(audio_path)}")
            logger.info(f"Input tokens: {len(input_ids)}, Label tokens: {len(labels)}")
            
        # Log token structure information
            logger.info(f"=== Zero-Shot Example Structure (ID: {item_id}, Speaker: {speaker}) ===")
            logger.info(f"Reference Audio: {os.path.basename(audio_path)}")
            logger.info(f"Reference Text: {reference_text}")
            logger.info(f"Target Audio: {os.path.basename(target_audio_path)}")
            logger.info(f"Target Text: {target_text}")
            
            # Log token counts and structure
            ref_audio_len = len(reference_codes)
            target_audio_len = len(target_codes)
            
            logger.info(f"Token Structure:")
            logger.info(f"  - start_of_human: {start_of_human}")
            logger.info(f"  - reference_text_ids: {len(reference_text_ids)} tokens")
            logger.info(f"  - end_of_text: {end_of_text}, end_of_human: {end_of_human}")
            logger.info(f"  - start_of_ai: {start_of_ai}, start_of_speech: {start_of_speech}")
            logger.info(f"  - reference_audio: {ref_audio_len} tokens")
            logger.info(f"  - end_of_speech: {end_of_speech}, end_of_ai: {end_of_ai}")
            logger.info(f"  - start_of_human: {start_of_human}")
            logger.info(f"  - target_text_ids: {len(target_text_ids)} tokens")
            logger.info(f"  - end_of_text: {end_of_text}, end_of_human: {end_of_human}")
            
            # Log total lengths
            logger.info(f"Input IDs Length: {len(input_ids)}")
            logger.info(f"Labels Length: {len(labels)}")
            
            # Log first few and last few tokens of input_ids and labels
            max_tokens_to_show = 10
            
            # Show beginning of input_ids
            input_start = input_ids[:min(max_tokens_to_show, len(input_ids))]
            logger.info(f"Input IDs (first {len(input_start)}): {input_start}")
            
            # Show end of input_ids
            if len(input_ids) > max_tokens_to_show:
                input_end = input_ids[-min(max_tokens_to_show, len(input_ids)):]
                logger.info(f"Input IDs (last {len(input_end)}): {input_end}")
            
            # Show beginning of labels
            labels_start = labels[:min(max_tokens_to_show, len(labels))]
            logger.info(f"Labels (first {len(labels_start)}): {labels_start}")
            
            # Show the AI response part in labels (which is not in input_ids)
            if len(labels) > len(input_ids):
                ai_response_start = labels[len(input_ids):len(input_ids)+min(max_tokens_to_show, len(labels)-len(input_ids))]
                logger.info(f"Labels AI Response (first {len(ai_response_start)}): {ai_response_start}")
            
            # Show end of labels
            if len(labels) > max_tokens_to_show:
                labels_end = labels[-min(max_tokens_to_show, len(labels)):]
                logger.info(f"Labels (last {len(labels_end)}): {labels_end}")
            
            # Verify that input_ids is a prefix of labels
            is_prefix = all(a == b for a, b in zip(input_ids, labels[:len(input_ids)]))
            logger.info(f"Input IDs is prefix of Labels: {is_prefix}")
            
            logger.info("=" * 50)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "speaker": speaker,  # Add speaker information for statistics
            "reference_audio": os.path.basename(audio_path),
            "target_audio": os.path.basename(target_audio_path)
        }
    except Exception as e:
        # Changed from debug to avoid excessive logs
        return None

def read_manifest_in_chunks(manifest_path, chunk_size=100000):
    """Read manifest file in chunks to avoid loading everything into memory"""
    with open(manifest_path, 'r') as f:
        chunk = []
        skipped_files = 0
        for i, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line)
                    # Pre-check if audio file exists to avoid loading invalid items
                    audio_path = item.get("audio_filepath")
                    if not audio_path or not os.path.exists(audio_path):
                        skipped_files += 1
                        if skipped_files % 10000 == 0:  # Reduced frequency from 1000 to 10000
                            logger.warning(f"Skipped {skipped_files} non-existent audio files so far")
                        continue
                        
                    chunk.append(item)
                    if len(chunk) >= chunk_size:
                        logger.info(f"Yielding chunk with {len(chunk)} valid items")
                        yield chunk
                        chunk = []
                except json.JSONDecodeError:
                    # Removed debug log
                    pass
        
        if chunk:  # Don't forget the last chunk
            logger.info(f"Yielding final chunk with {len(chunk)} valid items")
            yield chunk

def build_speaker_utterances_map(manifest_path):
    """Build a map of speaker to utterances for efficient pairing"""
    speaker_utterances = defaultdict(list)
    
    logger.info(f"Building speaker utterances map from {manifest_path}")
    with open(manifest_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line)
                    speaker = item.get("speaker", item.get("voice_name", "unknown_speaker"))
                    # Only filter out invalid audio paths
                    audio_path = item.get("audio_filepath")
                    if audio_path and os.path.exists(audio_path):
                        speaker_utterances[speaker].append(item)
                except json.JSONDecodeError:
                    continue
    
    # Only filter speakers with single utterance (required for pairing)
    filtered_speakers = {speaker: utterances for speaker, utterances in speaker_utterances.items() 
                        if len(utterances) > 1}
    
    logger.info(f"Found {len(filtered_speakers)} speakers with multiple utterances")
    return filtered_speakers

def process_chunk_parallel(items_chunk, manifest_items, speaker_utterances, num_workers):
    """Process a chunk of items in parallel using ThreadPoolExecutor"""
    if num_workers is None or num_workers <= 1:
        # Process sequentially
        processed_data = []
        for item in tqdm(items_chunk):
            result = process_item_zero_shot(item, manifest_items, speaker_utterances)
            if result:
                processed_data.append(result)
        return processed_data
    
    # Process in parallel with optimized chunk size:
    processed_data = []
    # Calculate optimal chunk size based on number of items and workers
    chunk_size = max(1, len(items_chunk) // (num_workers * 4))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use map with chunking for better load balancing
        futures = []
        for item in items_chunk:
            futures.append(executor.submit(process_item_zero_shot, item, manifest_items, speaker_utterances))
        
        # Collect results
        for future in tqdm(futures, total=len(futures), desc="Processing items"):
            result = future.result()
            if result:
                processed_data.append(result)
    
    return processed_data

def log_dataset_statistics(dataset_path):
    """Log statistics about the processed dataset to verify correctness"""
    try:
        dataset = Dataset.load_from_disk(dataset_path)
        
        # Basic statistics
        logger.info(f"=== Dataset Statistics ===")
        logger.info(f"Total examples: {len(dataset)}")
        
        # Sample a few examples for detailed inspection
        num_samples = min(3, len(dataset))  # Reduced from 5 to 3
        logger.info(f"Examining {num_samples} random samples:")
        
        for i, idx in enumerate(random.sample(range(len(dataset)), num_samples)):
            example = dataset[idx]
            
            input_ids = example["input_ids"]
            labels = example["labels"]
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Input IDs length: {len(input_ids)}")
            logger.info(f"  Labels length: {len(labels)}")
            
            # Check if labels properly extend input_ids
            is_prefix = all(a == b for a, b in zip(input_ids, labels[:len(input_ids)]))
            logger.info(f"  Input IDs is prefix of Labels: {is_prefix}")
            
            # Calculate average lengths
            if i == 0:
                total_input_len = len(input_ids)
                total_labels_len = len(labels)
            else:
                total_input_len += len(input_ids)
                total_labels_len += len(labels)
        
        # Calculate and log averages
        if num_samples > 0:
            avg_input_len = total_input_len / num_samples
            avg_labels_len = total_labels_len / num_samples
            logger.info(f"Average input_ids length: {avg_input_len:.2f}")
            logger.info(f"Average labels length: {avg_labels_len:.2f}")
            logger.info(f"Average target audio length: {(avg_labels_len - avg_input_len):.2f}")
        
        logger.info("=" * 30)  # Reduced separator length
        return True
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        return False

def prepare_zero_shot_dataset(manifest_path, output_dir, config_path="config.yaml", push_to_hub=False, hub_name=None, num_workers=None, batch_size=1000):
    """Convert a local manifest file to a dataset for zero-shot voice cloning training."""
    global config, model, tokenizer
    
    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Enable CUDA optimization
        torch.backends.cudnn.benchmark = True
        
        # Initialize model and tokenizer in main process only
        logger.info(f"Loading SNAC model: {config.get('snac_model', 'hubertsiuzdak/snac_24khz')}")
        try:
            model = SNAC.from_pretrained(config.get('snac_model', 'hubertsiuzdak/snac_24khz')).eval().to("cuda:0")
        except Exception as e:
            logger.error(f"Error loading SNAC model: {str(e)}")
            raise
        
        logger.info(f"Loading tokenizer: {config['tokenizer_name']}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config['tokenizer_name'],
                pad_token_id=config.get("pad_token", 128264)
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
        
        # Build speaker utterances map for efficient pairing
        speaker_utterances = build_speaker_utterances_map(manifest_path)
        
        # Save speaker statistics before processing
        save_speaker_statistics(speaker_utterances, output_dir)
        
        # Read all manifest items for pairing (this is memory intensive but necessary for zero-shot)
        logger.info("Reading all manifest items for pairing")
        all_manifest_items = []
        with open(manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        all_manifest_items.append(item)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Read {len(all_manifest_items)} total manifest items")
        
        # Process in chunks
        total_processed = 0
        part_counter = 0
        
        # Initialize validation stats properly
        validation_stats = {
            "total_processed": 0,
            "valid_examples": 0,
            "invalid_examples": 0
        }
        
        # Determine optimal number of workers if not specified
        if num_workers is None:
            num_workers = min(os.cpu_count(), 16)  # Default to reasonable number
        
        logger.info(f"Using {num_workers} workers for processing")
        
        # Process manifest in chunks
        for chunk_idx, items_chunk in enumerate(read_manifest_in_chunks(manifest_path, batch_size)):
            logger.info(f"Processing chunk {chunk_idx+1} with {len(items_chunk)} items")
            
            # Update total processed count
            validation_stats["total_processed"] += len(items_chunk)
            
            # Process items in parallel or sequentially
            processed_data = process_chunk_parallel(items_chunk, all_manifest_items, speaker_utterances, num_workers)
            
            if processed_data:
                # Update valid examples count
                validation_stats["valid_examples"] += len(processed_data)
                
                # Save this chunk as a dataset part
                part_counter += 1
                part_path = os.path.join(output_dir, f"part_{part_counter}")
                logger.info(f"Saving part {part_counter} with {len(processed_data)} examples to {part_path}")
                
                # Verify data structure before saving
                if len(processed_data) > 0:
                    sample_item = processed_data[0]
                    expected_keys = ["input_ids", "labels", "attention_mask"]
                    if not all(key in sample_item for key in expected_keys):
                        logger.warning(f"Data structure validation failed. Missing keys in sample item: {[k for k in expected_keys if k not in sample_item]}")
                
                # Create and save dataset
                dataset = Dataset.from_list(processed_data)
                
                # Verify dataset before saving
                if len(dataset) != len(processed_data):
                    logger.warning(f"Dataset size mismatch: {len(dataset)} vs {len(processed_data)}")
                
                # Save to disk with verification
                dataset.save_to_disk(part_path)
                
                # Verify saved dataset
                try:
                    loaded_dataset = Dataset.load_from_disk(part_path)
                    if len(loaded_dataset) != len(dataset):
                        logger.warning(f"Saved dataset size mismatch: {len(loaded_dataset)} vs {len(dataset)}")
                    else:
                        logger.info(f"Successfully verified saved dataset part {part_counter}")
                except Exception as e:
                    logger.error(f"Error verifying saved dataset part {part_counter}: {str(e)}")
                
                total_processed += len(processed_data)
                
                # Clear memory
                del processed_data
                del dataset
                gc.collect()
                torch.cuda.empty_cache()
            else:
                logger.warning(f"No valid examples in chunk {chunk_idx+1}")
            
            # Update invalid examples count
            validation_stats["invalid_examples"] = validation_stats["total_processed"] - validation_stats["valid_examples"]
            
            # Periodically clear CUDA cache to prevent memory fragmentation
            if chunk_idx % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Merge all parts
        if part_counter > 0:
            logger.info(f"Merging {part_counter} dataset parts...")
            
            # Check if all parts exist before merging
            missing_parts = []
            for i in range(1, part_counter + 1):
                part_path = os.path.join(output_dir, f"part_{i}")
                if not os.path.exists(part_path):
                    missing_parts.append(i)
            
            if missing_parts:
                logger.warning(f"Missing dataset parts: {missing_parts}")
            
            # Load and merge parts in batches to avoid OOM
            merged_dataset = None
            batch_size = min(5, part_counter)  # Merge at most 5 parts at a time
            
            for i in range(1, part_counter + 1, batch_size):
                end_idx = min(i + batch_size - 1, part_counter)
                logger.info(f"Loading and merging parts {i} to {end_idx}")
                
                # Only load parts that exist
                parts = []
                for j in range(i, end_idx + 1):
                    part_path = os.path.join(output_dir, f"part_{j}")
                    if os.path.exists(part_path):
                        try:
                            part_dataset = Dataset.load_from_disk(part_path)
                            parts.append(part_dataset)
                        except Exception as e:
                            logger.error(f"Error loading part {j}: {str(e)}")
                
                if parts:
                    batch_merged = concatenate_datasets(parts)
                    
                    if merged_dataset is None:
                        merged_dataset = batch_merged
                    else:
                        merged_dataset = concatenate_datasets([merged_dataset, batch_merged])
                    
                    # Clean up to save memory
                    del parts
                    del batch_merged
                    gc.collect()
                    torch.cuda.empty_cache()
            
            if merged_dataset:
                # Save merged dataset
                merged_path = os.path.join(output_dir, "merged")
                logger.info(f"Saving merged dataset with {len(merged_dataset)} examples to {merged_path}")
                merged_dataset.save_to_disk(merged_path)
                
                # Verify merged dataset
                try:
                    loaded_merged = Dataset.load_from_disk(merged_path)
                    if len(loaded_merged) != len(merged_dataset):
                        logger.warning(f"Merged dataset size mismatch: {len(loaded_merged)} vs {len(merged_dataset)}")
                    else:
                        logger.info(f"Successfully verified merged dataset with {len(loaded_merged)} examples")
                except Exception as e:
                    logger.error(f"Error verifying merged dataset: {str(e)}")
                
                # Log dataset statistics to verify correctness
                logger.info("Analyzing dataset to verify pipeline correctness...")
                log_dataset_statistics(merged_path)
                
                # Push to hub if requested
                if push_to_hub and hub_name:
                    logger.info(f"Pushing dataset to hub: {hub_name}")
                    merged_dataset.push_to_hub(hub_name)
                
                # Clean up part files
                for i in range(1, part_counter + 1):
                    part_path = os.path.join(output_dir, f"part_{i}")
                    if os.path.exists(part_path):
                        try:
                            shutil.rmtree(part_path, ignore_errors=True)
                        except Exception as e:
                            logger.warning(f"Could not remove part directory {part_path}: {str(e)}")
                            # Try to remove files individually if directory removal fails
                            try:
                                for root, dirs, files in os.walk(part_path, topdown=False):
                                    for file in files:
                                        try:
                                            os.remove(os.path.join(root, file))
                                        except:
                                            pass
                                    for dir in dirs:
                                        try:
                                            os.rmdir(os.path.join(root, dir))
                                        except:
                                            pass
                                # Try one more time to remove the directory
                                try:
                                    os.rmdir(part_path)
                                except:
                                    logger.warning(f"Could not completely clean up {part_path}, but continuing anyway")
                            except:
                                pass
                
                logger.info(f"Zero-shot dataset preparation complete. Total examples: {len(merged_dataset)}")
                
                # Calculate and log validation rate
                if validation_stats["total_processed"] > 0:
                    validation_rate = (validation_stats["valid_examples"] / validation_stats["total_processed"]) * 100
                    logger.info(f"Validation rate: {validation_rate:.2f}% ({validation_stats['valid_examples']}/{validation_stats['total_processed']})")
                
                return merged_path
            else:
                logger.error("No valid dataset parts found")
                return None
        else:
            logger.error("No valid examples processed")
            return None
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Prepare zero-shot voice cloning dataset from manifest file")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--push_to_hub", action="store_true", help="Push dataset to Hugging Face Hub")
    parser.add_argument("--hub_name", type=str, help="Name for dataset on Hub")
    parser.add_argument("--num_workers", type=int, help="Number of worker processes")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    
    args = parser.parse_args()
    
    prepare_zero_shot_dataset(
        args.manifest,
        args.output_dir,
        args.config,
        args.push_to_hub,
        args.hub_name,
        args.num_workers,
        args.batch_size
    )

if __name__ == "__main__":
    main()





