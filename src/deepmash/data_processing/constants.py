TARGET_SR = 16000              # think its hard to go lower than 16kHz
CHUNK_DURATION_SEC = 15        # all chunks will be exactly this long
MIN_CHUNK_DURATION_SEC = 5     # discard chunks shorter than this (otherwise zero-pad to CHUNK_DURATION_SEC)
VOCAL_RMS_THRESHHOLD = 10      # discard chunks with vocal energy below this threshold

# mel-spectrogram settings
N_MELS = 64
F_MIN = 60 
F_MAX = 7800
NFFT = 1024        # >= WINDOW_SIZE, and should be power of 2 for efficiency
WINDOW_SIZE = 1024 # 64ms @ 16kHz 
HOP_SIZE = 320     # 20ms @ 16kHz
