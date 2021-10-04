import json
import os
import numpy as np
import torch
from collections import defaultdict
from pytorch_sound.utils.calculate import db2log
from pytorch_sound.settings import MIN_DB
from torch.utils.data import Dataset as TorchDataset
from .text import text_to_sequence
from .utils.tools import pad_1D, pad_2D


class Dataset(TorchDataset):
    def __init__(
        self, filename, preprocessed_path, text_cleaners, batch_size,
            pitch_min: float, energy_min: float, sort=False, drop_last=False,
            mel_log_min: float = db2log(MIN_DB),
            is_reference: bool = False
    ):
        self.preprocessed_path = preprocessed_path
        self.cleaners = text_cleaners
        self.batch_size = batch_size
        self.mel_log_min = mel_log_min
        self.pitch_min = pitch_min
        self.energy_min = energy_min
        self.is_reference = is_reference

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, 'speakers.json')) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        if self.is_reference:
            self.file_spk_map = defaultdict(list)
            for basename, spk in zip(self.basename, self.speaker):
                self.file_spk_map[spk].append(basename)

    def get_ref_mel(self, idx):
        # target speaker
        speaker = self.speaker[idx]
        ref_base_name = np.random.choice(self.file_spk_map[speaker], 1)[0]
        mel_path = os.path.join(
            self.preprocessed_path,
            'mel',
            '{}-mel-{}.npy'.format(speaker, ref_base_name),
        )
        mel = np.load(mel_path)
        return mel

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            'mel',
            '{}-mel-{}.npy'.format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            'pitch',
            '{}-pitch-{}.npy'.format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            'energy',
            '{}-energy-{}.npy'.format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            'duration',
            '{}-duration-{}.npy'.format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            'id': basename,
            'speaker': speaker_id,
            'text': phone,
            'raw_text': raw_text,
            'mel': mel,
            'pitch': pitch,
            'energy': energy,
            'duration': duration,
        }

        if self.is_reference:
            sample['ref_mel'] = self.get_ref_mel(idx)
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), 'r', encoding='utf-8'
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip('\n').split('|')
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]['id'] for idx in idxs]
        speakers = [data[idx]['speaker'] for idx in idxs]
        texts = [data[idx]['text'] for idx in idxs]
        raw_texts = [data[idx]['raw_text'] for idx in idxs]
        mels = [data[idx]['mel'] for idx in idxs]
        pitches = [data[idx]['pitch'] for idx in idxs]
        energies = [data[idx]['energy'] for idx in idxs]
        durations = [data[idx]['duration'] for idx in idxs]
        
        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels, self.mel_log_min)
        # mels = pad_2D(mels, 0)
        pitches = pad_1D(pitches, 0)
        energies = pad_1D(energies, 0)
        durations = pad_1D(durations)

        result = [
            ids,
            raw_texts,
            torch.from_numpy(speakers).long(),
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(mels).float(),
            torch.from_numpy(mel_lens),
            max(mel_lens),
            torch.from_numpy(pitches).float(),
            torch.from_numpy(energies),
            torch.from_numpy(durations).long()
        ]

        if self.is_reference:
            ref_mels = [data[idx]['ref_mel'] for idx in idxs]
            ref_mels = pad_2D(ref_mels, self.mel_log_min)
            # ref_mels = pad_2D(ref_mels, 0)
            result.append(torch.from_numpy(ref_mels))

        return result

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d['text'].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config['preprocessing']['text']['text_cleaners']

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config['path']['preprocessed_path'], 'speakers.json'
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip('\n').split('|')
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == '__main__':
    # Test
    import yaml
    from torch.utils.data import DataLoader

    preprocess_config = yaml.load(
        open('./config/LJSpeech/preprocess.yaml', 'r'), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open('./config/LJSpeech/train.yaml', 'r'), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        'train.txt', preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        'val.txt', preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['optimizer']['batch_size'] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['optimizer']['batch_size'],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            n_batch += 1
    print(
        'Training set  with size {} is composed of {} batches.'.format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            n_batch += 1
    print(
        'Validation set  with size {} is composed of {} batches.'.format(
            len(val_dataset), n_batch
        )
    )
