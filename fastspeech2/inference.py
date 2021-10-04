import re
import torch
import yaml
import numpy as np
from g2p_en import G2p
from fastspeech2.models.fastspeech2 import FastSpeech2, fast_speech2_vctk, fast_speech2_vctk_gst
from .text import text_to_sequence


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


class Inferencer:

    def __init__(self, chk_path: str, lexicon_path: str, device: str = 'cuda', is_gst: bool = False):
        self.chk_path = chk_path
        self.lexicon = read_lexicon(lexicon_path)
        self.g2p = G2p()
        self.cleaners = ['english_cleaners']
        self.device = device
        if not torch.cuda.is_available():
            self.device = 'cpu'

        # load model
        if is_gst:
            model_config = fast_speech2_vctk_gst
        else:
            model_config = fast_speech2_vctk
        self.model = FastSpeech2(**model_config()).to(device)
        self.model.eval()
        chkpt = torch.load(self.chk_path)
        self.model.load_state_dict(chkpt['model'])

    def process_txt(self, text):
        # make phones
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in self.lexicon:
                phones += self.lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", self.g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

        # make as numpy array
        sequence = np.array(text_to_sequence(phones, self.cleaners))
        return sequence

    @torch.no_grad()
    def tts(self, text: str, speaker: int = 0,
            pitch_control: float = 1., energy_control: float = 1., duration_control: float = 1.):
        # preprocess
        sequence = self.process_txt(text)
        txt_len = len(sequence)
        # make tensors
        spk_tensor = torch.LongTensor([speaker]).to(self.device)
        txt_tensor = torch.LongTensor([sequence]).to(self.device)
        txt_len_tensor = torch.LongTensor([txt_len]).to(self.device)

        with torch.no_grad():
            # forward tts
            raugh_mel, post_mel, *_ = self.model(
                spk_tensor,
                txt_tensor,
                txt_len_tensor,
                txt_len,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )

        return post_mel
