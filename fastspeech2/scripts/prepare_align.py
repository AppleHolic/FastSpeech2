import fire
import json
from fastspeech2.preprocessing import vctk


def run(dataset_name: str, config_path: str):
    with open(config_path, 'r') as r:
        config = json.load(r)
    if dataset_name == 'vctk':
        vctk.prepare_align(**config)
    else:
        raise NotImplementedError(f'{dataset_name} is not implemented! you should choose in vctk or libri_tts')


if __name__ == '__main__':
    fire.Fire(run)
