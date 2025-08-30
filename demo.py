"""
CSM Interactive Demo
Test conversational voice generation in real-time
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import time
from typing import Optional, List, Dict
import argparse

# For audio playback
try:
    import pyaudio
    AUDIO_PLAYBACK = True
except ImportError:
    AUDIO_PLAYBACK = False
    print("PyAudio not installed. Audio will be saved to files instead.")

from architecture import CSMModel
from rvq_tokenizer import ConversationalRVQ
from missing_components import TextTokenizer


class CSMDemo:
    """Interactive demo for CSM model"""
    
    def __init__(self, checkpoint_path: str = "checkpoints/test_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', self._default_config())
        
        # Initialize models
        self.model = CSMModel(
            d_model=self.config['model']['d_model'],
            n_layers=self.config['model']['n_layers'],
            n_heads=self.config['model']['n_heads']
        ).to(self.device)
        
        self.rvq = ConversationalRVQ(
            n_codebooks=self.config['rvq']['n_codebooks'],
            codebook_size=self.config['rvq']['codebook_size']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state'])
        self.rvq.load_state_dict(checkpoint['rvq_state'])
        
        self.model.eval()
        self.rvq.eval()
        
        # Text tokenizer
        self.tokenizer = TextTokenizer()
        
        # Conversation history
        self.conversation_history = []
        
        # Audio setup
        self.sample_rate = 24000
        if AUDIO_PLAYBACK:
            self.audio = pyaudio.PyAudio()
            self.stream = None
        
        print("Model loaded successfully!")
    
    def _default_config(self):
        """Default config if not in checkpoint"""
        return {
            'model': {'d_model': 768, 'n_layers': 12, 'n_heads': 12},
            'rvq': {'n_codebooks': 32, 'codebook_size': 1024}
        }
    
    def generate_response(
        self,
        text: str,
        speaker: str = "assistant",
        temperature: float = 0.8,
        max_length: int = 512
    ) -> Dict:
        """Generate voice response for text input"""
        
        start_time = time.time()
        
        # Tokenize input
        conversation_turns = self.conversation_history + [
            {'speaker': 'user', 'text': text}
        ]
        tokens = self.tokenizer.encode_conversation(
            conversation_turns,
            max_length=max_length
        )
        
        # Move to device
        input_ids = tokens['input_ids'].to(self.device)
        
        # Generate with model
        with torch.no_grad():
            # For now, we'll use teacher forcing with dummy audio
            # In full implementation, this would be autoregressive
            
            # Generate text continuation
            outputs = self.model(
                text_tokens=input_ids,
                conversation_history=self.conversation_history
            )
            
            # Sample next tokens
            text_logits = outputs['text_logits']
            
            # Simple sampling (would be more sophisticated in production)
            if temperature > 0:
                probs = torch.softmax(text_logits[:, -1, :] / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=20)
            else:
                next_tokens = text_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Generate voice codes
            # For demo, we'll create synthetic codes
            voice_codes = self._generate_voice_codes(
                text_features=outputs.get('text_hidden', input_ids),
                speaker=speaker,
                emotion="neutral"
            )
            
            # Decode to audio
            audio = self.rvq.decode(voice_codes)
        
        # Convert to numpy
        audio_np = audio[0, 0].cpu().numpy()  # First sample, first channel
        
        # Decode text response
        response_text = self.tokenizer.decode(next_tokens[0])
        
        generation_time = time.time() - start_time
        
        result = {
            'text': response_text,
            'audio': audio_np,
            'speaker': speaker,
            'generation_time': generation_time,
            'audio_duration': len(audio_np) / self.sample_rate
        }
        
        # Update conversation history
        self.conversation_history.append({
            'speaker': 'user',
            'text': text
        })
        self.conversation_history.append({
            'speaker': speaker,
            'text': response_text,
            'audio': audio_np
        })
        
        return result
    
    def _generate_voice_codes(
        self,
        text_features: torch.Tensor,
        speaker: str,
        emotion: str
    ) -> List[torch.Tensor]:
        """Generate voice codes from text features"""
        # In production, this would use the model's voice generation
        # For demo, create plausible codes
        
        batch_size = text_features.shape[0]
        seq_len = 240  # ~10ms frames for 2.4 seconds
        
        codes = []
        for i in range(self.config['rvq']['n_codebooks']):
            if i < 10:  # Semantic codebooks
                # Use text features to influence semantic codes
                code_logits = torch.randn(
                    batch_size, seq_len, self.config['rvq']['codebook_size'],
                    device=self.device
                )
                # Add text influence
                text_influence = text_features.mean(dim=1, keepdim=True)
                text_influence = text_influence.expand(-1, seq_len, -1)
                code_logits[:, :, :text_influence.shape[2]] += text_influence * 0.5
            else:  # Acoustic codebooks
                # Random acoustic variations
                code_logits = torch.randn(
                    batch_size, seq_len, self.config['rvq']['codebook_size'],
                    device=self.device
                )
            
            codes.append(code_logits.argmax(dim=-1))
        
        return codes
    
    def play_audio(self, audio: np.ndarray):
        """Play audio through speakers"""
        if AUDIO_PLAYBACK:
            # Normalize audio
            audio = np.clip(audio, -1, 1)
            audio = (audio * 32767).astype(np.int16)
            
            # Open stream if needed
            if self.stream is None:
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True
                )
            
            # Play audio
            self.stream.write(audio.tobytes())
        else:
            # Save to file
            timestamp = int(time.time())
            filename = f"outputs/audio/response_{timestamp}.wav"
            Path("outputs/audio").mkdir(parents=True, exist_ok=True)
            sf.write(filename, audio, self.sample_rate)
            print(f"Audio saved to {filename}")
    
    def interactive_session(self):
        """Run interactive conversation session"""
        print("\n" + "="*50)
        print("CSM Interactive Demo")
        print("Type your message and hear the AI respond!")
        print("Commands: 'quit' to exit, 'history' to see conversation")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'history':
                    self.print_history()
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                print("Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response['text'])
                
                # Play audio
                self.play_audio(response['audio'])
                
                # Show stats
                print(f"  [Generated in {response['generation_time']:.2f}s, "
                      f"Audio: {response['audio_duration']:.2f}s]")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")
        
        if AUDIO_PLAYBACK and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
    
    def print_history(self):
        """Print conversation history"""
        print("\n--- Conversation History ---")
        for turn in self.conversation_history:
            speaker = "You" if turn['speaker'] == 'user' else "Assistant"
            print(f"{speaker}: {turn['text']}")
        print("----------------------------\n")
    
    def batch_generate(self, conversations: List[Dict]) -> List[Dict]:
        """Generate responses for multiple conversations"""
        results = []
        
        for conv in conversations:
            # Reset history for each conversation
            self.conversation_history = conv.get('history', [])
            
            # Generate response
            response = self.generate_response(
                text=conv['text'],
                speaker=conv.get('speaker', 'assistant'),
                temperature=conv.get('temperature', 0.8)
            )
            
            results.append(response)
        
        return results


def run_test_examples(demo: CSMDemo):
    """Run test examples to showcase capabilities"""
    print("\n" + "="*50)
    print("Running test examples...")
    print("="*50 + "\n")
    
    test_cases = [
        {
            'name': 'Greeting',
            'text': 'Hello! How are you today?',
            'expected_quality': 'Friendly and warm'
        },
        {
            'name': 'Question',
            'text': 'What is the weather like?',
            'expected_quality': 'Informative with rising intonation'
        },
        {
            'name': 'Emotional',
            'text': "I'm feeling a bit sad today.",
            'expected_quality': 'Empathetic and gentle'
        },
        {
            'name': 'Excited',
            'text': "I just got promoted at work!",
            'expected_quality': 'Congratulatory and energetic'
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Input: {test['text']}")
        print(f"Expected: {test['expected_quality']}")
        
        response = demo.generate_response(test['text'])
        print(f"Response: {response['text']}")
        
        # Save audio
        filename = f"outputs/audio/test_{test['name'].lower()}.wav"
        Path("outputs/audio").mkdir(parents=True, exist_ok=True)
        sf.write(filename, response['audio'], demo.sample_rate)
        print(f"Audio saved to {filename}")
        
        # Play if available
        if AUDIO_PLAYBACK:
            demo.play_audio(response['audio'])
            time.sleep(0.5)  # Brief pause between examples


def main():
    parser = argparse.ArgumentParser(description='CSM Interactive Demo')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/test_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, 
                        choices=['interactive', 'test', 'batch'],
                        default='interactive',
                        help='Demo mode')
    parser.add_argument('--input', type=str,
                        help='Input file for batch mode')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found at {args.checkpoint}")
        print("Please train a model first using train_minimal.py")
        return
    
    # Create demo
    demo = CSMDemo(args.checkpoint)
    
    if args.mode == 'interactive':
        demo.interactive_session()
    elif args.mode == 'test':
        run_test_examples(demo)
    elif args.mode == 'batch':
        if not args.input:
            print("Batch mode requires --input file")
            return
        
        with open(args.input, 'r') as f:
            conversations = json.load(f)
        
        results = demo.batch_generate(conversations)
        
        # Save results
        output_file = args.input.replace('.json', '_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()