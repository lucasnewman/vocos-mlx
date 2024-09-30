from subprocess import CalledProcessError, run

import mlx.core as mx
import numpy as np

def load_audio(file: str, sr: int = 24_000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0

def save_audio(file: str, audio: mx.array, sampling_rate: int):
    """
    Save audio to a wave (.wav) file.
    """
    from scipy.io.wavfile import write

    audio = (audio * 32767).astype(mx.int16)
    write(file, sampling_rate, np.array(audio))
