import fire
import soundfile as sf
from fastspeech2.inference import Inferencer
from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN


def synthesize(text: str, out_path: str, chk_path: str, lexicon_path: str, device: str = 'cuda', speaker: int = 0,
               sample_rate: int = 22050):
    # save file path
    if not out_path.endswith('.wav'):
        raise Exception(f'{out_path} is not supported output type. You should define as a wav file.')

    # arguments
    # chk_path: str, lexicon_path: str, device: str = 'cuda'
    print('Load FastSpeech2 ...')
    inferencer = Inferencer(chk_path=chk_path, lexicon_path=lexicon_path, device=device)

    # arguments
    # text: str, speaker: int = 0, pitch_control: float = 1., energy_control: float = 1., duration_control: float = 1.
    print('Inference')
    mel_spectrogram = inferencer.tts(text, speaker=speaker)

    # Reconstructs speech by using Hifi-GAN
    print('Load HifiGAN v1 ...')
    interface = InterfaceHifiGAN(model_name='hifi_gan_v1_universal', device='cuda')
    pred_wav = interface.decode(mel_spectrogram.transpose(1, 2)).squeeze()

    # save file path
    sf.write(out_path, pred_wav.cpu().numpy(), samplerate=sample_rate, format='WAV')
    print('Finish!')


if __name__ == '__main__':
    fire.Fire(synthesize)
