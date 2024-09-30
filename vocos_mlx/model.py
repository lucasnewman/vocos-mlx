from __future__ import annotations
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Union
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from huggingface_hub import snapshot_download
import yaml

from vocos_mlx.encodec import EncodecModel


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Saved using extract_filterbank.py
    """
    assert n_mels in {100}, f"Unsupported n_mels: {n_mels}"

    filename = os.path.join("assets", "mel_filters.npz")
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
    filterbank: Optional[mx.array] = None,
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
    filters = filterbank if filterbank is not None else mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return mx.expand_dims(log_spec, axis=0)


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
        filterbank: Optional[mx.array] = None,
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.filterbank = filterbank

    def __call__(self, audio, **kwargs):
        return log_mel_spectrogram(
            audio,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            padding=0,
            filterbank=self.filterbank,
        )


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()

        if encodec_model == "encodec_24khz":
            encodec, preprocessor = EncodecModel.from_pretrained(
                "mlx-community/encodec-24khz-float32"
            )
        elif encodec_model == "encodec_48khz":
            encodec, preprocessor = EncodecModel.from_pretrained(
                "mlx-community/encodec-48khz-float32"
            )
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )

        self.encodec = encodec
        self.preprocessor = preprocessor
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            bandwidth=max(bandwidths)
        )
        self.codebook_weights = mx.concatenate(
            [vq.codebook.embed for vq in self.encodec.quantizer.layers[: self.num_q]]
        )
        self.bandwidths = bandwidths

    def get_encodec_codes(self, audio: mx.array, bandwidth_id: int) -> mx.array:
        features, mask = self.preprocessor(audio)
        codes, _ = self.encodec.encode(
            features, mask, bandwidth=self.bandwidths[bandwidth_id]
        )
        return mx.reshape(codes, (codes.shape[-2], 1, codes.shape[-1]))

    def get_features_from_codes(self, codes: mx.array) -> mx.array:
        offsets = mx.arange(
            0,
            self.encodec.quantizer.codebook_size * codes.shape[0],
            self.encodec.quantizer.codebook_size,
        )
        embeddings_idxs = codes + mx.reshape(offsets, (offsets.shape[0], 1, 1))
        embeddings = self.codebook_weights[embeddings_idxs]
        features = mx.sum(embeddings, axis=0)
        return features

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")

        codes = self.get_encodec_codes(audio, bandwidth_id=bandwidth_id)
        return self.get_features_from_codes(codes)


class ISTFTHead(nn.Module):
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


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()

        # depthwise conv
        self.dwconv = GroupableConv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = layer_scale_init_value * mx.ones(dim)

    def __call__(
        self, x: mx.array, cond_embedding_id: Optional[mx.array] = None
    ) -> mx.array:
        residual = x
        x = self.dwconv(x)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim

        self.scale = nn.Embedding(num_embeddings=num_embeddings, dims=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, dims=embedding_dim)
        self.scale.weight = mx.ones((num_embeddings, embedding_dim))
        self.shift.weight = mx.zeros((num_embeddings, embedding_dim))

    def __call__(self, x: mx.array, cond_embedding_id: mx.array) -> mx.array:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = mx.fast.layer_norm(x, weight=None, bias=None, eps=self.eps)
        x = x * scale + shift
        return x


class GroupableConv1d(nn.Module):
    """Applies a 1-dimensional convolution over the multi-channel input sequence.

    The channels are expected to be last i.e. the input shape should be ``NLC`` where:

    * ``N`` is the batch dimension
    * ``L`` is the sequence length
    * ``C`` is the number of input channels

    Args:
        in_channels (int): The number of input channels
        out_channels (int): The number of output channels
        kernel_size (int): The size of the convolution filters
        stride (int, optional): The stride when applying the filter.
            Default: ``1``.
        padding (int, optional): How many positions to 0-pad the input with.
            Default: ``0``.
        dilation (int, optional): The dilation of the convolution.
        groups (int, optional): The number of groups for the convolution.
            Default: ``1``.
        bias (bool, optional): If ``True`` add a learnable bias to the output.
            Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError(
                f"The number of input channels ({in_channels}) must be "
                f"divisible by the number of groups ({groups})"
            )

        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1]}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        y = mx.conv1d(
            x, self.weight, self.stride, self.padding, self.dilation, self.groups
        )
        if "bias" in self:
            y = y + self.bias
        return y


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = [
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                adanorm_num_embeddings=adanorm_num_embeddings,
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        bandwidth_id = kwargs.get("bandwidth_id", None)

        x = self.embed(x)

        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x, bandwidth_id)
        else:
            x = self.norm(x)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x)
        return x


class Vocos(nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: VocosBackbone,
        head: ISTFTHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(
        cls, config_path: str, filterbank: Optional[mx.array] = None
    ) -> Vocos:
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = SimpleNamespace(**yaml.safe_load(f))

        if "MelSpectrogramFeatures" in config.feature_extractor["class_path"]:
            feature_extractor_init_args = config.feature_extractor["init_args"]
            if filterbank is not None:
                feature_extractor_init_args["filterbank"] = filterbank
            feature_extractor = MelSpectrogramFeatures(**feature_extractor_init_args)
        elif "EncodecFeatures" in config.feature_extractor["class_path"]:
            feature_extractor = EncodecFeatures(**config.feature_extractor["init_args"])
        backbone = VocosBackbone(**config.backbone["init_args"])
        head = ISTFTHead(**config.head["init_args"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(cls, path_or_repo: str) -> Vocos:
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """

        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.yaml", "*.safetensors"],
                )
            )

        model_path = path / "model.safetensors"
        with open(model_path, "rb") as f:
            weights = mx.load(f)

        # load the filterbank for model initialization

        try:
            filterbank = weights.pop(
                "feature_extractor.mel_spec.mel_scale.fb"
            ).moveaxis(0, 1)
        except KeyError:
            filterbank = None

        config_path = path / "config.yaml"
        model = cls.from_hparams(config_path, filterbank)

        # remove unused weights
        try:
            del weights["feature_extractor.mel_spec.spectrogram.window"]
            del weights["head.istft.window"]
        except KeyError:
            pass

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

        # use strict = False to avoid the encodec weights
        model.load_weights(list(new_weights.items()), strict=False)
        model.eval()

        return model

    def __call__(self, audio_input: mx.array, **kwargs: Any) -> mx.array:
        features = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    def get_encodec_codes(self, audio_input: mx.array, bandwidth_id: int) -> mx.array:
        if not isinstance(self.feature_extractor, EncodecFeatures):
            raise ValueError("This model does not support getting encodec codes.")

        return self.feature_extractor.get_encodec_codes(audio_input, bandwidth_id)

    def decode(self, features_input: mx.array, **kwargs: Any) -> mx.array:
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def decode_from_codes(self, codes: mx.array, **kwargs: Any) -> mx.array:
        features = self.feature_extractor.get_features_from_codes(codes)
        audio_output = self.decode(features, **kwargs)
        return audio_output
