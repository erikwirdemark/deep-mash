TARGET_SR = 16000           # think its hard to go lower than 16kHz
CHUNK_DURATION_SEC = 15     # all chunks will be exactly this long
MIN_CHUNK_DURATION_SEC = 5  # discard chunks shorter than this (otherwise zero-pad to CHUNK_DURATION_SEC)

# mel-spectrogram settings (using same as cocola for now)
N_MELS = 64
F_MIN = 60 
F_MAX = 7800
WINDOW_SIZE = 1024 # 64ms @ 16kHz (should be power of 2 for efficiency)
HOP_SIZE = 320     # 20ms @ 16kHz
