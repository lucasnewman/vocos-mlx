import librosa
import numpy as np

filterbank = librosa.filters.mel(
    sr=24000,
    n_fft=1024,
    n_mels=100,
    norm = None,
    htk = True
)

np.savez_compressed(
    "assets/mel_filters.npz",
    mel_100=filterbank
)
