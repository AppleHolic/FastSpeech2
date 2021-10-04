import fire
import json
import inspect
from fastspeech2.preprocessing.preprocessor import Preprocessor
from fastspeech2.utils.tools import parse_kwargs


def run(config_path: str):
    # load json config
    with open(config_path, 'r') as r:
        config = json.load(r)

    config = parse_kwargs(Preprocessor, **config)

    # make preprocessor
    preprocessor = Preprocessor(**config)
    preprocessor.build_from_path()


if __name__ == '__main__':
    fire.Fire(run)
