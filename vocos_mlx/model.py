from __future__ import annotations

import os
from functools import lru_cache

from typing import Any, Optional, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn

import yaml

from huggingface_hub import hf_hub_download


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using extract_filterbank.py
    """
    assert n_mels in {100}, f"Unsupported n_mels: {n_mels}"

    filename = os.path.join("assets", "mel_filters.npz")
    print(f"Loading filterbank: {filename}")
    return mx.load(filename, format="npz")[f"mel_{n_mels}"]


@lru_cache(maxsize=None)
def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])


def stft(x, window, nperseg=256, noverlap=None, nfft=None, pad_mode="constant"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def istft(x, window, nperseg=256, noverlap=None, nfft=None):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    t = (x.shape[0] - 1) * noverlap + nperseg
    reconstructed = mx.zeros(t)
    window_sum = mx.zeros(t)

    for i in range(x.shape[0]):
        # inverse FFT of each frame
        frame_time = mx.fft.irfft(x[i])

        # get the position in the time-domain signal to add the frame
        start = i * noverlap
        end = start + nperseg

        # overlap-add the inverse transformed frame, scaled by the window
        reconstructed[start:end] += frame_time * window
        window_sum[start:end] += window

    # normalize by the sum of the window values
    reconstructed = mx.where(window_sum != 0, reconstructed / window_sum, reconstructed)

    return reconstructed


def log_mel_spectrogram(
    audio: Union[mx.array, np.ndarray],
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array], shape = (*)
        The path to audio or either a NumPy or mlx array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 100 is supported

    padding: int
        Number of zero samples to pad to the right

    Returns
    -------
    mx.array, shape = (n_mels, n_frames)
        An  array that contains the Mel spectrogram
    """
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)

    if padding > 0:
        audio = mx.pad(audio, (0, padding))

    freqs = stft(audio, hanning(n_fft), nperseg=n_fft, noverlap=hop_length)
    magnitudes = freqs[:-1, :].abs()
    filters = mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return log_spec


class ISTFTHeadMLX(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.out(x).swapaxes(1, 2)
        mag, p = x.split(2, axis=1)
        mag = mx.exp(mag)
        mag = mx.clip(mag, None, 1e2)
        x = mx.cos(p)
        y = mx.sin(p)
        S = mag * (x + 1j * y)
        audio = istft(
            S.squeeze(0).swapaxes(0, 1),
            hanning(self.n_fft),
            self.n_fft,
            self.hop_length,
            self.n_fft,
        )
        return audio


class ConvNeXtBlockMLX(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float):
        super().__init__()

        # depthwise conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = layer_scale_init_value * mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = residual + x
        return x


class VocosBackboneMLX(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = [
            ConvNeXtBlockMLX(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.embed(x)
        x = self.norm(x)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x)
        return x


class VocosMLX(nn.Module):
    def __init__(
        self,
        backbone: VocosBackboneMLX,
        head: ISTFTHeadMLX,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str) -> VocosMLX:
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        backbone = VocosBackboneMLX(**config["backbone"]["init_args"])
        head = ISTFTHeadMLX(**config["head"]["init_args"])
        model = cls(backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: Optional[str] = None) -> VocosMLX:
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.yaml", revision=revision
        )
        model = cls.from_hparams(config_path)

        model_path = hf_hub_download(
            repo_id=repo_id, filename="model.safetensors", revision=revision
        )
        weights = mx.load(model_path)

        # remove unused weights
        del weights["feature_extractor.mel_spec.spectrogram.window"]
        del weights["feature_extractor.mel_spec.mel_scale.fb"]
        del weights["head.istft.window"]

        # transpose weights as needed
        new_weights = {}
        for k, v in weights.items():
            basename, pname = k.rsplit(".", 1)
            if "backbone.embed" in basename and pname == "weight":
                new_weights[k] = v.moveaxis(1, 2)
            elif "dwconv" in basename and pname == "weight":
                new_weights[k] = v.moveaxis(1, 2)
            else:
                new_weights[k] = v

        model.load_weights(list(new_weights.items()))
        model.eval()

        return model

    def __call__(self, audio_input: mx.array, **kwargs: Any) -> mx.array:
        features = mx.expand_dims(log_mel_spectrogram(audio_input), axis=0)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    def decode(self, features_input: mx.array, **kwargs: Any) -> mx.array:
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output
