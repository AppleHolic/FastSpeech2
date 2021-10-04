import os
import random
import json
import librosa
import numpy as np
import pyworld as pw
import tgt
import torch
from multiprocessing import cpu_count
from typing import Any, Dict
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from speech_interface.encoders import MelSpectrogram


def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


def normalize(in_dir, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value


def get_alignment(tier, sample_rate, hop_size):
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sample_rate / hop_size)
                - np.round(s * sample_rate / hop_size)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def process_utterance(in_dir, out_dir, speaker, basename, sample_rate, hop_size, stft_func,
                      pitch_phoneme_averaging, energy_phoneme_averaging):
    wav_path = os.path.join(in_dir, speaker, '{}.wav'.format(basename))
    text_path = os.path.join(in_dir, speaker, '{}.lab'.format(basename))
    # tg_path = os.path.join(
    #     out_dir, 'TextGrid', speaker, '{}.TextGrid'.format(basename)
    # )
    tg_path = os.path.join(
        out_dir, speaker, '{}.TextGrid'.format(basename)
    )

    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name('phones'), sample_rate, hop_size
    )
    text = '{' + ' '.join(phone) + '}'
    if start >= end:
        return None

    # Read and trim wav files
    wav, _ = librosa.load(wav_path)
    wav = wav[
        int(sample_rate * start): int(sample_rate * end)
    ].astype(np.float32)

    # Read raw text
    with open(text_path, 'r') as f:
        raw_text = f.readline().strip('\n')

    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sample_rate,
        frame_period=hop_size / sample_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sample_rate)

    pitch = pitch[: sum(duration)]
    if np.sum(pitch != 0) <= 1:
        return None

    # Compute mel-scale spectrogram and energy
    magnitude = stft_func.stft(torch.FloatTensor(wav).unsqueeze(0))[0]
    mel_spectrogram = torch.log(stft_func.to_mel(magnitude))[0]
    energy = magnitude.norm(dim=1).numpy()

    mel_spectrogram = mel_spectrogram[:, :sum(duration)]
    energy = energy[0, : sum(duration)]

    if pitch_phoneme_averaging:
        # perform linear interpolation
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                pitch[i] = np.mean(pitch[pos : pos + d])
            else:
                pitch[i] = 0
            pos += d
        pitch = pitch[: len(duration)]

    if energy_phoneme_averaging:
        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                energy[i] = np.mean(energy[pos:pos + d])
            else:
                energy[i] = 0
            pos += d
        energy = energy[: len(duration)]

    # Save files
    dur_filename = '{}-duration-{}.npy'.format(speaker, basename)
    np.save(os.path.join(out_dir, 'duration', dur_filename), duration)

    pitch_filename = '{}-pitch-{}.npy'.format(speaker, basename)
    np.save(os.path.join(out_dir, 'pitch', pitch_filename), pitch)

    energy_filename = '{}-energy-{}.npy'.format(speaker, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy)

    mel_filename = '{}-mel-{}.npy'.format(speaker, basename)
    np.save(
        os.path.join(out_dir, 'mel', mel_filename),
        mel_spectrogram.T,
    )

    return (
        '|'.join([basename, speaker, text, raw_text]),
        remove_outlier(pitch),
        remove_outlier(energy),
        mel_spectrogram.shape[1],
    )


class Preprocessor:
    def __init__(self, in_dir: str, out_dir: str, validation_rate: float, audio_params: Dict[str, Any],
                 pitch_feature: str = 'phoneme_level', energy_feature: str = 'phoneme_level',
                 pitch_norm: bool = True, energy_norm: bool = True, sample_rate: int = 22050,
                 unaligned_file: str = ''):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.validation_rate = validation_rate
        self.sample_rate = sample_rate
        self.sample_rate = audio_params['sample_rate']
        self.hop_size = audio_params['hop_size']
        if unaligned_file:
            with open(unaligned_file, 'r') as r:
                self.unaligned_list = [l.strip().split()[0] for l in r.readlines()]
        else:
            self.unaligned_list = []

        assert pitch_feature in [
            'phoneme_level',
            'frame_level',
        ]
        assert energy_feature in [
            'phoneme_level',
            'frame_level',
        ]
        self.pitch_phoneme_averaging = (pitch_feature == 'phoneme_level')
        self.energy_phoneme_averaging = energy_feature == 'phoneme_level'

        self.pitch_normalization = pitch_norm
        self.energy_normalization = energy_norm

        self.stft_func = MelSpectrogram(**audio_params)

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, 'mel')), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, 'pitch')), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, 'energy')), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, 'duration')), exist_ok=True)

        print('Processing Data ...')
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}

        with Parallel(n_jobs=cpu_count() - 1) as parallel:

            for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
                speakers[speaker] = i

                basenames = [os.path.basename(wav_name).split('.')[0] for wav_name
                             in os.listdir(os.path.join(self.in_dir, speaker))
                             if wav_name.endswith('.wav')]
                if self.unaligned_list:
                    basenames = [name for name in basenames if name not in self.unaligned_list]
                results = parallel(
                    delayed(process_utterance)
                    (self.in_dir, self.out_dir, speaker, basename, self.sample_rate, self.hop_size, self.stft_func,
                     self.pitch_phoneme_averaging, self.energy_phoneme_averaging)
                    for basename in basenames
                )
                for res in results:
                    if res is None:
                        continue
                    info, pitch, energy, n = res
                    out.append(info)
                    n_frames += n

                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

        print('Computing statistic quantities ...')
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = normalize(
            os.path.join(self.out_dir, 'pitch'), pitch_mean, pitch_std
        )
        energy_min, energy_max = normalize(
            os.path.join(self.out_dir, 'energy'), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, 'speakers.json'), 'w') as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, 'stats.json'), 'w') as f:
            stats = {
                'pitch': [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                'energy': [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            'Total time: {} hours'.format(
                n_frames * self.hop_size / self.sample_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]
        val_size = int(self.validation_rate * len(out))

        # Write metadata
        with open(os.path.join(self.out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            for m in out[val_size:]:
                f.write(m + '\n')
        with open(os.path.join(self.out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
            for m in out[:val_size]:
                f.write(m + '\n')

        return out
