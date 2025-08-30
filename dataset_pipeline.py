"""
Conversational Audio Dataset Pipeline
Philosophy: Conversations are not just sequences of utterances - they're dynamic interactions
with context, emotion, and purpose.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os
from pathlib import Path
import webdataset as wds
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf


@dataclass
class ConversationalTurn:
    """Single turn in a conversation"""
    speaker_id: str
    text: str
    audio_path: Optional[str]
    audio_array: Optional[np.ndarray]
    start_time: float
    end_time: float
    emotion: Optional[str]
    context_turns: List['ConversationalTurn']
    prosodic_features: Optional[Dict[str, float]]
    
    def to_dict(self) -> dict:
        return {
            'speaker_id': self.speaker_id,
            'text': self.text,
            'audio_path': self.audio_path,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'emotion': self.emotion,
            'prosodic_features': self.prosodic_features
        }


class ProsodyAnalyzer:
    """Extract prosodic features that matter for conversation"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        
    def extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract conversationally relevant prosodic features"""
        features = {}
        
        # Pitch features (fundamental for emotion and emphasis)
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=self.sample_rate
        )
        
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            features['pitch_mean'] = np.mean(f0_voiced)
            features['pitch_std'] = np.std(f0_voiced)
            features['pitch_range'] = np.ptp(f0_voiced)
            
            # Pitch dynamics (how pitch changes over time)
            pitch_delta = np.diff(f0_voiced)
            features['pitch_dynamism'] = np.std(pitch_delta) if len(pitch_delta) > 0 else 0
        else:
            features.update({
                'pitch_mean': 0, 'pitch_std': 0,
                'pitch_range': 0, 'pitch_dynamism': 0
            })
        
        # Energy features (volume dynamics indicate emphasis)
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        features['energy_mean'] = np.mean(energy)
        features['energy_std'] = np.std(energy)
        features['energy_range'] = np.ptp(energy)
        
        # Speaking rate (approximated by zero crossing rate)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['speaking_rate'] = np.mean(zcr)
        
        # Spectral features (voice quality)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['voice_brightness'] = np.mean(spectral_centroids)
        
        # Pause patterns (silence ratio - important for conversation flow)
        silence_threshold = 0.01 * np.max(np.abs(audio))
        silence_frames = np.abs(audio) < silence_threshold
        features['silence_ratio'] = np.mean(silence_frames)
        
        return features


class ConversationSegmenter:
    """Segment continuous audio into conversational turns"""
    
    def __init__(
        self,
        vad_threshold: float = 0.5,
        min_speech_duration: float = 0.5,
        max_speech_duration: float = 30.0,
        min_silence_duration: float = 0.3
    ):
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.min_silence_duration = min_silence_duration
        
        # Would use a proper VAD model in production
        # For now, energy-based segmentation
        
    def segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcript: Optional[List[Dict]] = None
    ) -> List[Tuple[float, float]]:
        """Segment audio into speech turns"""
        
        # If we have transcript with timestamps, use those
        if transcript and all('start' in t and 'end' in t for t in transcript):
            return [(t['start'], t['end']) for t in transcript]
        
        # Otherwise, use energy-based VAD
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)     # 10ms hop
        
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Adaptive threshold
        energy_threshold = np.percentile(energy, 30)
        
        # Find speech regions
        speech_frames = energy > energy_threshold
        
        # Convert to time segments
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(speech_frames):
            time = i * hop_length / sample_rate
            
            if is_speech and not in_speech:
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                duration = time - start_time
                if duration >= self.min_speech_duration:
                    # Split long segments
                    if duration > self.max_speech_duration:
                        n_splits = int(duration / self.max_speech_duration) + 1
                        split_duration = duration / n_splits
                        for j in range(n_splits):
                            segments.append((
                                start_time + j * split_duration,
                                start_time + (j + 1) * split_duration
                            ))
                    else:
                        segments.append((start_time, time))
                in_speech = False
        
        # Handle final segment
        if in_speech:
            segments.append((start_time, len(audio) / sample_rate))
        
        return segments


class ConversationalDataset(Dataset):
    """
    Dataset for conversational audio
    Key innovation: Maintains conversational context and speaker continuity
    """
    
    def __init__(
        self,
        data_paths: List[str],
        sample_rate: int = 24000,
        max_duration: float = 10.0,
        context_window: int = 5,
        augment: bool = True
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.context_window = context_window
        self.augment = augment
        
        self.prosody_analyzer = ProsodyAnalyzer(sample_rate)
        self.segmenter = ConversationSegmenter()
        
        # Load metadata
        self.conversations = []
        for path in data_paths:
            self._load_conversations(path)
        
        # Build turn index for efficient sampling
        self.turn_index = []
        for conv_idx, conversation in enumerate(self.conversations):
            for turn_idx in range(len(conversation['turns'])):
                self.turn_index.append((conv_idx, turn_idx))
    
    def _load_conversations(self, path: str):
        """Load conversation data from various formats"""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.conversations.extend(data)
                else:
                    self.conversations.append(data)
        elif path.suffix == '.tar':
            # WebDataset format
            dataset = wds.WebDataset(str(path))
            for sample in dataset:
                self.conversations.append(self._parse_wds_sample(sample))
        else:
            # Directory of audio files
            for audio_file in path.glob('**/*.wav'):
                self.conversations.append(self._create_single_turn_conversation(audio_file))
    
    def _parse_wds_sample(self, sample: dict) -> dict:
        """Parse WebDataset sample into conversation format"""
        return {
            'conversation_id': sample.get('__key__', 'unknown'),
            'turns': sample.get('turns', []),
            'speakers': sample.get('speakers', {}),
            'metadata': sample.get('metadata', {})
        }
    
    def _create_single_turn_conversation(self, audio_path: Path) -> dict:
        """Create conversation dict from single audio file"""
        return {
            'conversation_id': audio_path.stem,
            'turns': [{
                'speaker_id': 'speaker_0',
                'audio_path': str(audio_path),
                'text': '',  # Would need ASR or transcript
                'start_time': 0,
                'end_time': None  # Will be computed when loaded
            }],
            'speakers': {'speaker_0': {'name': 'Unknown'}},
            'metadata': {}
        }
    
    def __len__(self) -> int:
        return len(self.turn_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conv_idx, turn_idx = self.turn_index[idx]
        conversation = self.conversations[conv_idx]
        turn = conversation['turns'][turn_idx]
        
        # Load audio
        audio, sr = self._load_audio(turn)
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Get context turns
        context_start = max(0, turn_idx - self.context_window)
        context_turns = conversation['turns'][context_start:turn_idx]
        
        # Extract features
        prosodic_features = self.prosody_analyzer.extract_features(audio)
        
        # Load context audio and features
        context_audio = []
        context_features = []
        for ctx_turn in context_turns[-3:]:  # Last 3 turns for memory
            ctx_audio, ctx_sr = self._load_audio(ctx_turn)
            if ctx_sr != self.sample_rate:
                ctx_audio = librosa.resample(ctx_audio, orig_sr=ctx_sr, target_sr=self.sample_rate)
            context_audio.append(ctx_audio)
            context_features.append(self.prosody_analyzer.extract_features(ctx_audio))
        
        # Prepare output
        output = {
            'audio': torch.from_numpy(audio).float(),
            'text': turn.get('text', ''),
            'speaker_id': turn['speaker_id'],
            'prosodic_features': prosodic_features,
            'context_audio': [torch.from_numpy(ca).float() for ca in context_audio],
            'context_features': context_features,
            'context_speakers': [ct['speaker_id'] for ct in context_turns[-3:]],
            'conversation_id': conversation['conversation_id']
        }
        
        # Apply augmentation if training
        if self.augment and np.random.random() < 0.5:
            output = self._augment(output)
        
        return output
    
    def _load_audio(self, turn: dict) -> Tuple[np.ndarray, int]:
        """Load audio for a turn"""
        if 'audio_array' in turn and turn['audio_array'] is not None:
            return turn['audio_array'], self.sample_rate
        
        audio_path = turn['audio_path']
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Extract segment if timestamps provided
        if turn.get('start_time') is not None and turn.get('end_time') is not None:
            start_sample = int(turn['start_time'] * sr)
            end_sample = int(turn['end_time'] * sr)
            audio = audio[start_sample:end_sample]
        
        return audio, sr
    
    def _augment(self, sample: dict) -> dict:
        """Apply conversational augmentations"""
        audio = sample['audio'].numpy()
        
        # Speed perturbation (simulate different speaking rates)
        if np.random.random() < 0.3:
            speed_factor = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        # Pitch shift (simulate different speakers)
        if np.random.random() < 0.3:
            pitch_shift = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=pitch_shift
            )
        
        # Add reverb (simulate different environments)
        if np.random.random() < 0.2:
            audio = self._add_reverb(audio)
        
        sample['audio'] = torch.from_numpy(audio).float()
        return sample
    
    def _add_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Simple reverb effect"""
        delay_samples = int(0.05 * self.sample_rate)
        decay = 0.3
        
        reverb = np.zeros_like(audio)
        reverb[delay_samples:] = audio[:-delay_samples] * decay
        
        return audio + reverb


class ConversationalCollator:
    """
    Custom collator that handles variable-length conversational data
    """
    
    def __init__(self, max_audio_length: int = 240000):  # 10 seconds at 24kHz
        self.max_audio_length = max_audio_length
    
    def __call__(self, batch: List[dict]) -> dict:
        # Find max lengths
        max_audio_len = min(
            self.max_audio_length,
            max(sample['audio'].shape[0] for sample in batch)
        )
        
        # Prepare batched tensors
        batch_size = len(batch)
        audio_batch = torch.zeros(batch_size, 1, max_audio_len)
        
        # Collect other data
        texts = []
        speaker_ids = []
        prosodic_features = []
        
        for i, sample in enumerate(batch):
            # Pad or truncate audio
            audio_len = min(sample['audio'].shape[0], max_audio_len)
            audio_batch[i, 0, :audio_len] = sample['audio'][:audio_len]
            
            texts.append(sample['text'])
            speaker_ids.append(sample['speaker_id'])
            prosodic_features.append(sample['prosodic_features'])
        
        return {
            'audio': audio_batch,
            'audio_lengths': torch.tensor([s['audio'].shape[0] for s in batch]),
            'text': texts,
            'speaker_ids': speaker_ids,
            'prosodic_features': prosodic_features,
            # Context is complex - keep as list for now
            'context': [s.get('context_audio', []) for s in batch]
        }


def create_dataloader(
    data_paths: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """Create DataLoader for conversational audio"""
    
    dataset = ConversationalDataset(data_paths, **dataset_kwargs)
    collator = ConversationalCollator()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )


# Specialized datasets for different data sources
class PodcastDataset(ConversationalDataset):
    """Specialized dataset for podcast data"""
    
    def _load_conversations(self, path: str):
        """Load podcast episodes with speaker diarization"""
        # Podcasts have natural conversations but need diarization
        # Would integrate with pyannote or similar
        super()._load_conversations(path)


class AudiobookDataset(ConversationalDataset):
    """Specialized dataset for audiobook data"""
    
    def __init__(self, *args, single_speaker: bool = False, **kwargs):
        self.single_speaker = single_speaker
        super().__init__(*args, **kwargs)
    
    def _load_conversations(self, path: str):
        """Load audiobook with character dialogue extraction"""
        # Audiobooks have rich emotional content
        # Would need NLP to extract dialogue vs narration
        super()._load_conversations(path)


if __name__ == "__main__":
    # Test the dataset pipeline
    
    # Create dummy data
    dummy_conversation = {
        'conversation_id': 'test_001',
        'turns': [
            {
                'speaker_id': 'alice',
                'text': 'Hello, how are you?',
                'audio_array': np.random.randn(24000),  # 1 second
                'start_time': 0,
                'end_time': 1,
                'emotion': 'neutral'
            },
            {
                'speaker_id': 'bob',
                'text': "I'm doing great, thanks!",
                'audio_array': np.random.randn(36000),  # 1.5 seconds
                'start_time': 1.5,
                'end_time': 3,
                'emotion': 'happy'
            }
        ],
        'speakers': {
            'alice': {'name': 'Alice', 'gender': 'F'},
            'bob': {'name': 'Bob', 'gender': 'M'}
        }
    }
    
    # Save dummy data
    os.makedirs('test_data', exist_ok=True)
    with open('test_data/test_conversation.json', 'w') as f:
        json.dump([dummy_conversation], f)
    
    # Create dataset
    dataset = ConversationalDataset(['test_data/test_conversation.json'])
    
    # Test loading
    sample = dataset[0]
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Speaker: {sample['speaker_id']}")
    print(f"Text: {sample['text']}")
    print(f"Prosodic features: {list(sample['prosodic_features'].keys())}")
    
    # Test dataloader
    dataloader = create_dataloader(['test_data/test_conversation.json'], batch_size=2)
    
    for batch in dataloader:
        print(f"Batch audio shape: {batch['audio'].shape}")
        print(f"Batch speakers: {batch['speaker_ids']}")
        break