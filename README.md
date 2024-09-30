# Vocos — MLX 

Implementation of [Vocos](https://github.com/gemelo-ai/vocos) with the [MLX](https://github.com/ml-explore/mlx) framework.

### Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

## Installation

To use Vocos in inference mode, install it using:

```bash
pip install vocos-mlx
```

## Usage

### Mel Spectrogram

```python
from vocos_mlx import Vocos, load_audio, log_mel_spectrogram

vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")

# reconstruct
audio = load_audio("audio.wav", 24_000)
reconstructed_audio = vocos(audio)

# decode from mel spec
mel_spec = log_mel_spectrogram(audio, n_mels = 100)
decoded_audio = vocos.decode(mel_spec)
```

### Encodec

```python
from vocos_mlx import Vocos, load_audio

vocos = Vocos.from_pretrained("lucasnewman/vocos-encodec-24khz")

# reconstruct
audio = load_audio("audio.wav", 24_000)
reconstructed_audio = vocos(audio, bandwidth_id = 3)

# decode with encodec codes
codes = vocos.feature_extractor.get_encodec_codes(audio, bandwidth_id = 3)
decoded_audio = vocos.decode_from_codes(codes, bandwidth_id = 3)
```

## Citations

```
@article{siuzdak2023vocos,
  title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis},
  author={Siuzdak, Hubert},
  journal={arXiv preprint arXiv:2306.00814},
  year={2023}
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
