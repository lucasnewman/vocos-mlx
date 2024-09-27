# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

Implementation of [Vocos](https://github.com/gemelo-ai/vocos) with the [MLX](https://github.com/ml-explore/mlx) framework.

Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

## Installation

To use Vocos in inference mode, install it using:

```bash
pip install vocos-mlx
```

## Usage

```python
from vocos_mlx import Vocos, load_audio, log_mel_spectrogram

audio = load_audio("audio.wav", 24_000)
mel_spec = log_mel_spectrogram(audio, n_mels = 100)

vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")

# reconstruct
reconstructed_audio = vocos_mlx(audio)

# decode
decoded_audio = vocos_mlx.decode(mx.expand_dims(mel_spec, axis=0))
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
