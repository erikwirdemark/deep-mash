import os
import numpy as np
import librosa
import time
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DEBUG FUNCTIONS
# ============================================================================

import time
import matplotlib.pyplot as plt

def debug_plot_spectrogram(spec, title="Spectrogram"):
    """Optional quick visual debug for spectrograms."""
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ============================================================================
# 1. AUDIO PROCESSING UTILITIES
# ============================================================================

class AudioProcessor:
    """Handles all audio loading, mixing, and spectrogram computation."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def load_audio(self, file_path: Path) -> np.ndarray:
        print(f"[DEBUG] Loading audio: {file_path}")
        t0 = time.time()
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        print(f"[DEBUG] Loaded {len(audio)} samples (sr={sr}) in {time.time()-t0:.2f}s")
        return audio

    # Don't know the correct way of combining the separate instrument stems into one: 
    # Just summing them for now and normalizing
    def mix_stems(self, stem_paths: List[Path]) -> np.ndarray:
        print(f"[DEBUG] Mixing stems: {[str(p.name) for p in stem_paths]}")
        t0 = time.time()
        stems = [self.load_audio(path) for path in stem_paths]
        min_len = min(len(stem) for stem in stems)
        stems = [stem[:min_len] for stem in stems]
        mixed = sum(stems)
        max_val = np.abs(mixed).max()
        if max_val > 0:
            mixed = mixed / max_val * 0.9
        print(f"[DEBUG] Mixed in {time.time()-t0:.2f}s | length={len(mixed)}")
        return mixed

    def extract_segment(
        self,
        audio: np.ndarray,
        duration: float,
        offset: Optional[float] = None
    ) -> np.ndarray:
        """Extract fixed-duration segment from audio."""
        target_length = int(duration * self.sample_rate)
        
        if len(audio) < target_length:
            # Pad if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # Extract segment
            if offset is not None:
                start = int(offset * self.sample_rate)
            else:
                max_start = len(audio) - target_length
                start = np.random.randint(0, max(1, max_start))
            audio = audio[start:start + target_length]
        
        return audio
    
    def compute_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute normalized log mel spectrogram."""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        log_mel_spec = (log_mel_spec + 80) / 80
        log_mel_spec = np.clip(log_mel_spec, 0, 1)
        
        return log_mel_spec

class AudioAugmentor:
    """Handles audio augmentation for training."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Random time stretch (±10%)
        if np.random.rand() > 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Random pitch shift (±2 semitones)
        if np.random.rand() > 0.5:
            n_steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=n_steps
            )
        
        # Random gain (±3 dB)
        if np.random.rand() > 0.5:
            gain_db = np.random.uniform(-3, 3)
            audio = audio * (10 ** (gain_db / 20))
        
        return audio


# ============================================================================
# 2. DATASET DISCOVERY
# ============================================================================

class StemDiscovery:
    """Discovers and validates GTZAN stem files."""
    
    @staticmethod
    def find_tracks(data_dir: Path, original_tracks_dir: Optional[Path] = None) -> List[Dict]:
        """
        Find all valid tracks with required stems.
        Expected structure: 
          - data_dir/genre/track_name/*.wav (stems)
          - original_tracks_dir/genre/track_name.wav (original mix - optional)
        
        Args:
            data_dir: Path to GTZAN stems directory
            original_tracks_dir: Path to original (unseparated) tracks (optional)
        
        Returns:
            List of track dictionaries with stem paths and optional original track path
        """
        tracks = []
        genre_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        for genre_dir in genre_dirs:
            track_dirs = [d for d in genre_dir.iterdir() if d.is_dir()]
            
            for track_dir in track_dirs:
                stems = StemDiscovery._find_stems_in_directory(track_dir)
                
                # Check if all required stems exist (vocals, drums, bass, other)
                if not all(v is not None for v in stems.values()):
                    continue
                
                # Find original mixed track (optional)
                original_path = None
                if original_tracks_dir is not None:
                    original_path = StemDiscovery._find_original_track(
                        original_tracks_dir, genre_dir.name, track_dir.name
                    )
                
                tracks.append({
                    'track_name': track_dir.name,
                    'genre': genre_dir.name,
                    'stems': stems,
                    'original': original_path
                })
        
        return tracks
    
    @staticmethod
    def _find_stems_in_directory(track_dir: Path) -> Dict[str, Optional[Path]]:
        """Find stem files in a track directory."""
        stems = {
            'vocals': None,
            'drums': None,
            'bass': None,
            'other': None
        }
        
        for stem_file in track_dir.glob('*.wav'):
            stem_name = stem_file.stem.lower()
            if 'vocal' in stem_name:
                stems['vocals'] = stem_file
            elif 'drum' in stem_name:
                stems['drums'] = stem_file
            elif 'bass' in stem_name:
                stems['bass'] = stem_file
            elif 'other' in stem_name or 'accomp' in stem_name:
                stems['other'] = stem_file
        
        return stems
    
    @staticmethod
    def _find_original_track(
        original_dir: Path,
        genre: str,
        track_name: str
    ) -> Optional[Path]:
        """Find the original (unseparated) track."""
        # Try different possible locations
        possible_paths = [
            original_dir / genre / f"{track_name}.wav",
            original_dir / genre / f"{track_name}.mp3",
            original_dir / f"{track_name}.wav",
            original_dir / f"{track_name}.mp3",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None

# ============================================================================
# 3. DATASET CLASS
# ============================================================================

class GTZANStemsDataset(Dataset):
    """
    Dataset for vocal-instrumental matching from GTZAN Stems.
    
    The model learns to match vocals with instrumentals, using the original
    mixed track as ground truth reference for what the correct pairing sounds like.
    
    Returns:
        - Vocal spectrogram (isolated vocal stem)
        - Instrumental spectrogram (drums + bass + other mixed, NO vocals)
        - Original track spectrogram (ground truth: vocals + instrumentals properly mixed)
        - Track index (for positive pair identification)
    """
    
    def __init__(
        self,
        stems_dir: str,
        original_tracks_dir: str,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        duration: float = 10.0,
        segment_offset: Optional[float] = None,
        augment: bool = True,
    ):
        """
        Args:
            stems_dir: Path to GTZAN stems directory
            original_tracks_dir: Path to original (unseparated) tracks directory
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            duration: Duration of audio segments in seconds
            segment_offset: Fixed offset for reproducibility (None = random)
            augment: Whether to apply data augmentation
        """
        self.stems_dir = Path(stems_dir)
        self.original_tracks_dir = Path(original_tracks_dir)
        self.duration = duration
        self.segment_offset = segment_offset
        self.augment = augment
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.augmentor = AudioAugmentor(sample_rate=sample_rate) if augment else None
        
        # Discover tracks (requires both stems AND original tracks)
        self.tracks = StemDiscovery.find_tracks(
            self.stems_dir,
            self.original_tracks_dir
        )
        
        # Filter out tracks without original audio
        self.tracks = [t for t in self.tracks if t['original'] is not None]
        
        print(f"Found {len(self.tracks)} tracks with stems and original ground truth")

    
    def __len__(self) -> int:
        return len(self.tracks)
    
    def __getitem__(self, idx: int):
        start_time = time.time()
        print(f"\n[DEBUG] Loading item {idx}/{len(self.tracks)} ...")

        track = self.tracks[idx]
        print(f"[DEBUG] Track: {track['track_name']} | Genre: {track['genre']}")

        try:
            # Process vocal
            print("[DEBUG] Loading vocal stem...")
            vocal = self._process_vocal(track)
            print(f"[DEBUG] Vocal loaded: {vocal.shape}")

            # Process instrumental
            print("[DEBUG] Loading instrumental stems...")
            instrumental = self._process_instrumental(track)
            print(f"[DEBUG] Instrumental loaded: {instrumental.shape}")

            # Process original
            print("[DEBUG] Loading original mix...")
            original = self._process_original(track)
            print(f"[DEBUG] Original loaded: {original.shape}")

            # Compute spectrograms
            print("[DEBUG] Computing spectrograms...")
            vocal_spec = self.audio_processor.compute_melspectrogram(vocal)
            instrumental_spec = self.audio_processor.compute_melspectrogram(instrumental)
            original_spec = self.audio_processor.compute_melspectrogram(original)
            print(f"[DEBUG] Specs shapes: vocal={vocal_spec.shape}, inst={instrumental_spec.shape}, orig={original_spec.shape}")

            # Optional quick plot (only for first few samples)
            if idx < 2:
                debug_plot_spectrogram(vocal_spec, f"Vocal Spec - {track['track_name']}")
                debug_plot_spectrogram(instrumental_spec, f"Instrumental Spec - {track['track_name']}")
                debug_plot_spectrogram(original_spec, f"Original Spec - {track['track_name']}")

            # Convert to tensors
            vocal_spec = torch.FloatTensor(vocal_spec).unsqueeze(0)
            instrumental_spec = torch.FloatTensor(instrumental_spec).unsqueeze(0)
            original_spec = torch.FloatTensor(original_spec).unsqueeze(0)

            elapsed = time.time() - start_time
            print(f"[DEBUG] Finished item {idx} in {elapsed:.2f}s")

            return vocal_spec, instrumental_spec, original_spec, idx

        except Exception as e:
            print(f"[ERROR] Failed on track {track['track_name']}: {e}")
            raise

    
    def _process_vocal(self, track: Dict) -> np.ndarray:
        """Load and process vocal stem."""
        vocal = self.audio_processor.load_audio(track['stems']['vocals'])
        if self.augment:
            vocal = self.augmentor.augment(vocal)
         vocal = self.audio_processor.extract_segment(
            vocal, self.duration, self.segment_offset
        )
        return vocal
    
    def _process_instrumental(self, track: Dict) -> np.ndarray:
        """Mix and process instrumental stems (drums + bass + other, NO vocals)."""
        instrumental_stems = [
            track['stems']['drums'],
            track['stems']['bass'],
            track['stems']['other']
        ]
        instrumental = self.audio_processor.mix_stems(instrumental_stems)
        if self.augment:
            instrumental = self.augmentor.augment(instrumental)
        instrumental = self.audio_processor.extract_segment(
            instrumental, self.duration, self.segment_offset
        )
        return instrumental
    
    def _process_original(self, track: Dict) -> np.ndarray:
        """Load and process original mixed track (ground truth)."""
        original = self.audio_processor.load_audio(track['original'])
        if self.augment:
            original = self.augmentor.augment(original)
         original = self.audio_processor.extract_segment(
            original, self.duration, self.segment_offset
        )
        return original


# ============================================================================
# 4. DATALOADER CREATION
# ============================================================================

def collate_contrastive_batch(batch):
    """
    Collate function for contrastive learning with ground truth.
    
    Returns:
        vocal_specs: (batch_size, 1, n_mels, time_steps) - isolated vocal stems
        instrumental_specs: (batch_size, 1, n_mels, time_steps) - instrumental mix (NO vocals)
        original_specs: (batch_size, 1, n_mels, time_steps) - ground truth original tracks
        labels: (batch_size,) track indices for positive pair identification
    """
    vocal_specs = torch.stack([item[0] for item in batch])
    instrumental_specs = torch.stack([item[1] for item in batch])
    original_specs = torch.stack([item[2] for item in batch])
    labels = torch.LongTensor([item[3] for item in batch])
    
    return vocal_specs, instrumental_specs, original_specs, labels


def create_dataloaders(
    stems_dir: str,
    original_tracks_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    **dataset_kwargs
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        stems_dir: Path to GTZAN stems directory
        original_tracks_dir: Path to original (unseparated) tracks directory (ground truth)
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
        **dataset_kwargs: Additional arguments for GTZANStemsDataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    dataset = GTZANStemsDataset(stems_dir, original_tracks_dir, **dataset_kwargs)
    
    print(f"[DEBUG] Total discovered tracks: {len(dataset.tracks)}")
    for i, t in enumerate(dataset.tracks[:5]):
        print(f"  [{i}] {t['genre']} - {t['track_name']}")

    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup paths
    stems_dir = r"D:\Users\ollet\Downloads\archive (1)\Data\genres_stems"  # Directory with separated stems
    original_tracks_dir = r"D:\Users\ollet\Downloads\archive (1)\Data\genres_original"  # Directory with original mixed tracks
    
    dataset = GTZANStemsDataset(stems_dir, original_tracks_dir, duration=5.0, augment=False)
    print(f"Dataset length: {len(dataset)}")

    print("\nTesting one sample manually:")
    vocal_spec, instrumental_spec, original_spec, idx = dataset[0]
    print(f"Shapes: {vocal_spec.shape}, {instrumental_spec.shape}, {original_spec.shape}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        stems_dir=stems_dir,
        original_tracks_dir=original_tracks_dir,
        batch_size=1,
        num_workers=0,
        sample_rate=22050,
        duration=10.0,
        augment=True,
    )
    
    # Test loading a batch
    print("\nLoading a batch...")
    vocal_batch, instrumental_batch, original_batch, labels = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  Vocal stems: {vocal_batch.shape}")
    print(f"  Instrumental stems: {instrumental_batch.shape}")
    print(f"  Original tracks (ground truth): {original_batch.shape}")
    print(f"  Labels: {labels.shape}")
    
    print(f"\nTraining setup:")
    print(f"  - Model learns: vocal[i] + instrumental[i] → should match original[i]")
    print(f"  - Positive pairs: (vocal[i], instrumental[i]) from same track")
    print(f"  - Negative pairs: (vocal[i], instrumental[j]) from different tracks")
    print(f"  - Ground truth: original[i] shows what correct pairing sounds like")
    

