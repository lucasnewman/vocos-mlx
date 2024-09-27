from setuptools import setup, find_packages

setup(
    name = 'vocos-mlx',
    packages = find_packages(exclude=[]),
    version = '0.0.1',
    license='MIT',
    description = 'Vocos - MLX',
    author = 'Lucas Newman',
    author_email = 'lucasnewman@me.com',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/lucasnewman/vocos-mlx',
    keywords = [
        'artificial intelligence',
        'asr',
        'audio-generation,'
        'deep learning',
        'transformers',
        'text-to-speech'
    ],
    install_requires=[
        'huggingface_hub',
        'mlx>=0.17.3',
        'numpy',
        'torch>=2.0',
        'torchaudio>=2.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
