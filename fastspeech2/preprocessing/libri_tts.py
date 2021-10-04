import os
import librosa
import numpy as np
import pandas as pd
from typing import List
from scipy.io import wavfile
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from ..text import _clean_text


def worker(wav_path: str, text: str, out_dir: str, speaker: str,
           sampling_rate: int, cleaners=['engllish_cleaners']):
    text = _clean_text(text, cleaners)
    base_name = os.path.basename(wav_path).split('.')[0]

    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
    wav, _ = librosa.load(wav_path, sampling_rate)
    wav = wav / max(abs(wav)) * 32767.0

    wavfile.write(
        os.path.join(out_dir, speaker, '{}.wav'.format(base_name)),
        sampling_rate,
        wav.astype(np.int16),
    )
    with open(os.path.join(out_dir, speaker, '{}.lab'.format(base_name)), 'w') as f1:
        f1.write(text)
    return ''


def prepare_align(out_dir: str, sample_rate: int, cleaners: List, meta_dir: str):
    meta_path = os.path.join(meta_dir, 'all_meta.json')

    # make input list
    df = pd.read_json(meta_path)
    input_wav_list = df['audio_filename']
    input_txt_list = df['text']
    speakers = df['speaker']

    # do parallel
    Parallel(n_jobs=cpu_count() - 1)(
        delayed(worker)
        (wav_path, text, out_dir, str(spk), sample_rate, cleaners)
        for wav_path, text, spk in tqdm(zip(input_wav_list, input_txt_list, speakers), desc='do parallel')
    )
