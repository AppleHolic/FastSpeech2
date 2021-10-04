import os
import librosa
import numpy as np
import pandas as pd
import re
import glob
from typing import List
from scipy.io import wavfile
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from ..text import _clean_text


def work(input_wav_path, txt_file, out_dir, cleaners, sampling_rate, max_wav_value):
    # get base name
    base_name = os.path.basename(input_wav_path).split('.')[0]
    speaker = input_wav_path.split('/')[-2]

    # clean text
    with open(txt_file, 'r') as r:
        text = r.read().strip()

    text = _clean_text(text, cleaners)
    if re.match(r'[\w\.\w]+', text):
        text = '. '.join(text.split('.'))

    # mkdir
    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

    # load wav
    wav, _ = librosa.load(input_wav_path, sampling_rate)

    # make int16 wavform data
    wav = wav / max(abs(wav)) * 32767.0

    # write wavefile
    wavfile.write(
        os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
        sampling_rate,
        wav.astype(np.int16),
    )

    # write textfile
    with open(
        os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
        "w",
    ) as f1:
        f1.write(text)


def prepare_align(in_dir: str, out_dir: str, train_speakers: str,
                  sample_rate: int, max_wav_value: int, cleaners: List):
    # load train speakers
    with open(train_speakers, 'r') as r:
        train_spks = [l.strip() for l in r.readlines()]

    # List-up all wav files
    wav_files = glob.glob(f'{in_dir}/wav48/*/*.wav')

    # List-up text files
    txt_files = glob.glob(f'{in_dir}/**/*.txt', recursive=True)

    # Match wav and text files, filter speakers
    info = {os.path.basename(file_path).split('.')[0]: {'wav_file': file_path} for file_path in wav_files}
    for text_file_path in txt_files:
        key = os.path.basename(text_file_path).split('.')[0]
        # filter train spks
        if key.split('_')[0] not in train_spks:
            continue
        if key in info:
            info[key]['txt_file'] = text_file_path


    # re-write
    input_wav_list = []
    input_txt_list = []
    for files in info.values():
        if len(files) == 1:
            continue
        wav, txt_file = files['wav_file'], files['txt_file']
        input_wav_list.append(wav)
        input_txt_list.append(txt_file)

    # do parallel
    Parallel(n_jobs=cpu_count() - 1)(
        delayed(work)
        (wav_path, txt_file, out_dir, cleaners, sample_rate, max_wav_value)
        for wav_path, txt_file in tqdm(zip(input_wav_list, input_txt_list), desc='do parallel')
    )
