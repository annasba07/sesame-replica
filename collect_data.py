"""
Data Collection Script for Conversational Audio
Downloads and prepares publicly available conversational datasets
"""

import os
import json
import tarfile
import zipfile
from pathlib import Path
import requests
import torch
import torchaudio
from tqdm import tqdm
import subprocess
import shutil
from typing import List, Dict, Optional
import pandas as pd
from datasets import load_dataset
import soundfile as sf
import numpy as np


class ConversationalDataCollector:
    """Collect conversational audio from various public sources"""
    
    def __init__(self, output_dir: str = "./data/conversations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data sources
        self.sources = {
            "voxceleb": {
                "url": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
                "description": "Celebrity interview clips",
                "hours": 2000
            },
            "common_voice": {
                "dataset": "mozilla-foundation/common_voice_11_0",
                "description": "Crowdsourced voice data",
                "languages": ["en"],
                "hours": 2000
            },
            "librispeech": {
                "urls": [
                    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
                    "https://www.openslr.org/resources/12/test-clean.tar.gz"
                ],
                "description": "Audiobook recordings",
                "hours": 5
            },
            "tedlium": {
                "url": "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz",
                "description": "TED talk recordings",
                "hours": 452
            }
        }
        
    def collect_all(self, target_hours: int = 100):
        """Collect data from all sources up to target hours"""
        print(f"Starting data collection for {target_hours} hours...")
        
        collected_hours = 0
        
        # 1. Start with LibriSpeech (easiest to get)
        if collected_hours < target_hours:
            hours = self.collect_librispeech()
            collected_hours += hours
            print(f"Collected {hours} hours from LibriSpeech. Total: {collected_hours}")
        
        # 2. Common Voice (conversational snippets)
        if collected_hours < target_hours:
            hours = self.collect_common_voice(max_hours=target_hours - collected_hours)
            collected_hours += hours
            print(f"Collected {hours} hours from Common Voice. Total: {collected_hours}")
        
        # 3. Create synthetic conversations from single-speaker data
        if collected_hours < target_hours:
            hours = self.create_synthetic_conversations(
                max_hours=target_hours - collected_hours
            )
            collected_hours += hours
            print(f"Created {hours} hours of synthetic conversations. Total: {collected_hours}")
        
        print(f"Data collection complete! Total: {collected_hours} hours")
        
        # Create metadata
        self.create_metadata()
        
    def collect_librispeech(self) -> float:
        """Download LibriSpeech dev and test sets"""
        libri_dir = self.output_dir / "librispeech"
        libri_dir.mkdir(exist_ok=True)
        
        total_duration = 0
        
        for url in self.sources["librispeech"]["urls"]:
            filename = url.split("/")[-1]
            filepath = libri_dir / filename
            
            # Download if not exists
            if not filepath.exists():
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract
            print(f"Extracting {filename}...")
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(libri_dir)
            
            # Process audio files
            subset_name = filename.replace('.tar.gz', '')
            subset_dir = libri_dir / "LibriSpeech" / subset_name
            
            for audio_file in subset_dir.rglob("*.flac"):
                # Convert to wav and resample to 24kHz
                audio, sr = torchaudio.load(str(audio_file))
                
                if sr != 24000:
                    resampler = torchaudio.transforms.Resample(sr, 24000)
                    audio = resampler(audio)
                
                # Save as wav
                output_path = libri_dir / "processed" / f"{audio_file.stem}.wav"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(output_path), audio, 24000)
                
                # Track duration
                duration = audio.shape[1] / 24000
                total_duration += duration
        
        return total_duration / 3600  # Convert to hours
    
    def collect_common_voice(self, max_hours: float = 50) -> float:
        """Download Common Voice English subset"""
        cv_dir = self.output_dir / "common_voice"
        cv_dir.mkdir(exist_ok=True)
        
        print("Loading Common Voice dataset...")
        
        # Load dataset (this will download if not cached)
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_11_0",
                "en",
                split="train",
                streaming=True  # Stream to avoid downloading everything
            )
        except Exception as e:
            print(f"Error loading Common Voice: {e}")
            print("Skipping Common Voice collection")
            return 0
        
        total_duration = 0
        target_seconds = max_hours * 3600
        
        # Process samples
        for i, sample in enumerate(tqdm(dataset, desc="Processing Common Voice")):
            if total_duration >= target_seconds:
                break
            
            # Get audio
            audio_array = sample['audio']['array']
            sr = sample['audio']['sampling_rate']
            
            # Resample to 24kHz if needed
            if sr != 24000:
                audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, 24000)
                audio_tensor = resampler(audio_tensor)
                audio_array = audio_tensor.squeeze().numpy()
            
            # Save audio
            output_path = cv_dir / f"cv_{i:06d}.wav"
            sf.write(str(output_path), audio_array, 24000)
            
            # Save metadata
            metadata = {
                'text': sample['sentence'],
                'speaker_id': sample.get('client_id', 'unknown'),
                'duration': len(audio_array) / 24000
            }
            
            with open(cv_dir / f"cv_{i:06d}.json", 'w') as f:
                json.dump(metadata, f)
            
            total_duration += metadata['duration']
        
        return total_duration / 3600
    
    def create_synthetic_conversations(self, max_hours: float = 20) -> float:
        """Create synthetic conversations by combining single-speaker utterances"""
        synth_dir = self.output_dir / "synthetic_conversations"
        synth_dir.mkdir(exist_ok=True)
        
        print("Creating synthetic conversations...")
        
        # Gather all available audio clips
        all_clips = []
        
        # From LibriSpeech
        libri_processed = self.output_dir / "librispeech" / "processed"
        if libri_processed.exists():
            for audio_file in libri_processed.glob("*.wav"):
                all_clips.append({
                    'path': audio_file,
                    'source': 'librispeech',
                    'speaker': audio_file.stem.split('-')[0]  # Extract speaker ID
                })
        
        # From Common Voice
        cv_dir = self.output_dir / "common_voice"
        if cv_dir.exists():
            for audio_file in cv_dir.glob("*.wav"):
                json_file = audio_file.with_suffix('.json')
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                    all_clips.append({
                        'path': audio_file,
                        'source': 'common_voice',
                        'speaker': metadata.get('speaker_id', 'unknown'),
                        'text': metadata.get('text', '')
                    })
        
        if not all_clips:
            print("No clips available for synthetic conversation creation")
            return 0
        
        # Group by speaker
        from collections import defaultdict
        speakers = defaultdict(list)
        for clip in all_clips:
            speakers[clip['speaker']].append(clip)
        
        # Create conversations
        total_duration = 0
        target_seconds = max_hours * 3600
        conversation_id = 0
        
        # Simple conversation patterns
        conversation_templates = [
            ["greeting", "greeting_response", "question", "answer", "closing"],
            ["question", "answer", "follow_up", "clarification", "acknowledgment"],
            ["statement", "agreement", "elaboration", "question", "response"]
        ]
        
        while total_duration < target_seconds and len(speakers) >= 2:
            # Select two different speakers
            speaker_ids = list(speakers.keys())
            if len(speaker_ids) < 2:
                break
            
            speaker1_id = np.random.choice(speaker_ids)
            speaker2_id = np.random.choice([s for s in speaker_ids if s != speaker1_id])
            
            speaker1_clips = speakers[speaker1_id]
            speaker2_clips = speakers[speaker2_id]
            
            if not speaker1_clips or not speaker2_clips:
                continue
            
            # Create a conversation
            template = np.random.choice(conversation_templates)
            conversation_audio = []
            conversation_metadata = {
                'conversation_id': f'synth_{conversation_id:06d}',
                'turns': []
            }
            
            for i, turn_type in enumerate(template):
                # Alternate speakers
                if i % 2 == 0:
                    clips = speaker1_clips
                    speaker = f'speaker1_{speaker1_id}'
                else:
                    clips = speaker2_clips
                    speaker = f'speaker2_{speaker2_id}'
                
                if not clips:
                    break
                
                # Select a clip
                clip_idx = np.random.randint(len(clips))
                clip = clips[clip_idx]
                
                # Load audio
                audio, sr = torchaudio.load(str(clip['path']))
                if sr != 24000:
                    resampler = torchaudio.transforms.Resample(sr, 24000)
                    audio = resampler(audio)
                
                # Add to conversation
                conversation_audio.append(audio)
                
                # Add silence between turns (0.5-1.5 seconds)
                silence_duration = np.random.uniform(0.5, 1.5)
                silence = torch.zeros(1, int(24000 * silence_duration))
                conversation_audio.append(silence)
                
                # Add metadata
                conversation_metadata['turns'].append({
                    'speaker': speaker,
                    'turn_type': turn_type,
                    'duration': audio.shape[1] / 24000,
                    'text': clip.get('text', '')
                })
                
                # Remove used clip
                clips.pop(clip_idx)
            
            if len(conversation_audio) > 0:
                # Concatenate all audio
                full_audio = torch.cat(conversation_audio, dim=1)
                
                # Save conversation
                output_path = synth_dir / f"conversation_{conversation_id:06d}.wav"
                torchaudio.save(str(output_path), full_audio, 24000)
                
                # Save metadata
                with open(synth_dir / f"conversation_{conversation_id:06d}.json", 'w') as f:
                    json.dump(conversation_metadata, f, indent=2)
                
                # Update duration
                duration = full_audio.shape[1] / 24000
                total_duration += duration
                conversation_id += 1
                
                if conversation_id % 100 == 0:
                    print(f"Created {conversation_id} conversations, {total_duration/3600:.1f} hours")
        
        return total_duration / 3600
    
    def create_metadata(self):
        """Create overall metadata file"""
        metadata = {
            'total_conversations': 0,
            'total_duration_hours': 0,
            'sources': [],
            'speakers': set(),
            'data_splits': {
                'train': [],
                'val': [],
                'test': []
            }
        }
        
        # Scan all directories
        for source_dir in self.output_dir.iterdir():
            if source_dir.is_dir():
                source_metadata = {
                    'source': source_dir.name,
                    'conversations': 0,
                    'duration_hours': 0
                }
                
                # Count conversations and duration
                for audio_file in source_dir.rglob("*.wav"):
                    audio_info = torchaudio.info(str(audio_file))
                    duration = audio_info.num_frames / audio_info.sample_rate
                    source_metadata['duration_hours'] += duration / 3600
                    source_metadata['conversations'] += 1
                    
                    # Add to splits (80/10/10)
                    rand = np.random.random()
                    if rand < 0.8:
                        metadata['data_splits']['train'].append(str(audio_file))
                    elif rand < 0.9:
                        metadata['data_splits']['val'].append(str(audio_file))
                    else:
                        metadata['data_splits']['test'].append(str(audio_file))
                
                metadata['sources'].append(source_metadata)
                metadata['total_conversations'] += source_metadata['conversations']
                metadata['total_duration_hours'] += source_metadata['duration_hours']
        
        # Convert speakers set to list for JSON serialization
        metadata['speakers'] = list(metadata['speakers'])
        
        # Save metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nData collection summary:")
        print(f"Total conversations: {metadata['total_conversations']}")
        print(f"Total duration: {metadata['total_duration_hours']:.1f} hours")
        print(f"Train set: {len(metadata['data_splits']['train'])} files")
        print(f"Val set: {len(metadata['data_splits']['val'])} files")
        print(f"Test set: {len(metadata['data_splits']['test'])} files")


def main():
    """Run data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect conversational audio data')
    parser.add_argument('--output_dir', type=str, default='./data/conversations',
                        help='Output directory for collected data')
    parser.add_argument('--target_hours', type=int, default=10,
                        help='Target hours of audio to collect')
    parser.add_argument('--sources', nargs='+', default=['all'],
                        choices=['all', 'librispeech', 'common_voice', 'synthetic'],
                        help='Data sources to use')
    args = parser.parse_args()
    
    # Create collector
    collector = ConversationalDataCollector(args.output_dir)
    
    # Collect data
    if 'all' in args.sources:
        collector.collect_all(target_hours=args.target_hours)
    else:
        collected_hours = 0
        if 'librispeech' in args.sources:
            collected_hours += collector.collect_librispeech()
        if 'common_voice' in args.sources:
            collected_hours += collector.collect_common_voice(
                max_hours=args.target_hours - collected_hours
            )
        if 'synthetic' in args.sources:
            collected_hours += collector.create_synthetic_conversations(
                max_hours=args.target_hours - collected_hours
            )
        
        collector.create_metadata()


if __name__ == "__main__":
    main()