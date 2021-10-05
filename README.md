# FastSpeech2 

This repository is a refactored version from [ming024's own](https://github.com/ming024/FastSpeech2).
I focused on refactoring structure for fitting my cases and making parallel pre-processing codes.
And I wrote installation guide with the latest version of MFA(Montreal Force Aligner).

## Installation

- Tested on python 3.8, Ubuntu 20.04
  - Notice ! For installing MFA, you should install the miniconda.
  - If you run MFA under 16.04 or ealier version of Ubuntu, you will face a compile error.
- In your system
  - To install pyworld, run "sudo apt-get install python3.x-dev". (x is your python version).
  - To install sndfile, run "sudo apt-get install libsndfile-dev"
  - To use MFA, run "sudo apt-get install libopenblas-base"

- Install requirements

```
# install pytorch_sound
pip install git+https://github.com/appleholic/pytorch_sound
pip install -e .
```

- Download datasets
1. VCTK
   - Visit and download dataset from [https://datashare.is.ed.ac.uk/handle/10283/2651](https://datashare.is.ed.ac.uk/handle/10283/2651)
   - Move to "./data" and extract compressed file.
     - If you wanna save dataset to another directory, you must change the path of configuration files.
2. LibriTTS
   - To be updated


- Install MFA 
  - Visit and follow a guide that described in [MFA installation website](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html).
  - Additional installation
     - mfa thirdparty download
     - mfa download acoustic english

## Preprocess (VCTK case)

1. Prepare MFA

```
python fastspeech2/scripts/prepare_align.py configs/vctk_prepare_align.json
```

2. Run MFA for making alignments

```
# Define your the number of threads to run MFA at the last of a command. "-j [The number of threads]"
mfa align data/fastspeech2/vctk lexicons/librispeech-lexicon.txt english data/fastspeech2/vctk-pre -j 24
```

3. Feature preprocessing

```
python fastspeech2/scripts/preprocess.py configs/vctk_preprocess.json
```

## Train

1. Multi-speaker fastspeech2

```
python fastspeech2/scripts/train.py configs/fastspeech2_vctk_tts.json
```

- If you want to change the parameters of training FastSpeech2, check out the code and put the option to configuration file.
  - train code : fastspeech2/scripts/train.py
  - config : configs/fastspeech2_vctk_tts.json

2. Fastspeech2 with reference encoder (To be updated)


## Synthesize 

### Multi-spaker model 

- In a code 

```python
from fastspeech2.inference import Inferencer
from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN

# arguments
# chk_path: str, lexicon_path: str, device: str = 'cuda'
inferencer = Inferencer(chk_path=chk_path, lexicon_path=lexicon_path, device=device)

# initialize hifigan
interface = InterfaceHifiGAN(model_name='hifi_gan_v1_universal', device='cuda')

# arguments
# text: str, speaker: int = 0, pitch_control: float = 1., energy_control: float = 1., duration_control: float = 1.
txt = 'Hello, I am a programmer.'
mel_spectrogram = inferencer.tts(txt, speaker=0)

# Reconstructs speech by using Hifi-GAN
pred_wav = interface.decode(mel_spectrogram.transpose(1, 2)).squeeze()

# If you test on a jupyter notebook
from IPython.display import Audio
Audio(pred_wav.cpu().numpy(), rate=22050)
```

- In command line

```
python fastspeech2/scripts/synthesize.py [TEXT] [OUTPUT PATH] [CHECKPOINT PATH] [LEXICON PATH] [[DEVICE]] [[SPEAKER]]
```


### Reference encoder (not updated)

## Reference

- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
