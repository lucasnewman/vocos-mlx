from __future__ import annotations

import os
from functools import lru_cache

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn

import yaml

from huggingface_hub import hf_hub_download

from encodec import EncodecModel
import torch


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)

    if "." not in init["class_path"]:
        class_name = init["class_path"]
        args_class = globals()[class_name]
    else:
        class_module, class_name = init["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using extract_filterbank.py
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


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding

    def __call__(self, audio, **kwargs):
        return mx.expand_dims(
            log_mel_spectrogram(
                audio, n_mels=100, n_fft=1024, hop_length=256, padding=0
            ),
            axis=0,
        )


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()
        
        # TODO: Use MLX encodec model.
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )

        self.codebook_weights = mx.concatenate(
            [
                mx.array(vq.codebook.numpy())
                for vq in self.encodec.quantizer.vq.layers[: self.num_q]
            ]
        )
        self.bandwidths = bandwidths

    def get_encodec_codes(self, audio: mx.array) -> mx.array:
        audio = torch.Tensor(memoryview(audio)).unsqueeze(0).unsqueeze(0)

        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(
            emb, self.encodec.frame_rate, self.encodec.bandwidth
        )

        return mx.array(codes.numpy())

    def get_features_from_codes(self, codes: mx.array) -> mx.array:
        offsets = mx.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins
        )
        embeddings_idxs = codes + mx.reshape(offsets, (offsets.shape[0], 1, 1))
        embeddings = self.codebook_weights[embeddings_idxs]
        features = mx.sum(embeddings, axis=0)
        return features

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")

        self.encodec.eval()
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
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
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
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
    def from_hparams(cls, config_path: str) -> Vocos:
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # remap the class paths
        if "MelSpectrogramFeatures" in config["feature_extractor"]["class_path"]:
            config["feature_extractor"]["class_path"] = "MelSpectrogramFeatures"
        elif "EncodecFeatures" in config["feature_extractor"]["class_path"]:
            config["feature_extractor"]["class_path"] = "EncodecFeatures"
        config["backbone"]["class_path"] = "VocosBackbone"
        config["head"]["class_path"] = "ISTFTHead"

        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: Optional[str] = None) -> Vocos:
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
        try:
            del weights["feature_extractor.mel_spec.spectrogram.window"]
            del weights["feature_extractor.mel_spec.mel_scale.fb"]
        except KeyError:
            pass

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
        features = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    def decode(self, features_input: mx.array, **kwargs: Any) -> mx.array:
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def decode_from_codes(self, codes: mx.array, **kwargs: Any) -> mx.array:
        features = self.feature_extractor.get_features_from_codes(codes)
        audio_output = self.decode(features, **kwargs)
        return audio_output
