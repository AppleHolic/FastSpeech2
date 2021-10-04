import torch.nn as nn
from typing import Dict, Any
from .models import Encoder, Decoder
from .layers import PostNet
from .modules import VarianceAdaptor
from pytorch_sound.models import register_model, register_model_architecture
from fastspeech2.utils.tools import get_mask_from_lengths
from .style_embed import GST


@register_model('fast_speech2')
class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any],
                 variance_adapter_config: Dict[str, Any], postnet_config: Dict[str, Any],
                 mel_size: int, n_speakers: int, is_gst: bool):
        super(FastSpeech2, self).__init__()
        self.is_gst = is_gst

        self.encoder = Encoder(**encoder_config)
        self.variance_adaptor = VarianceAdaptor(**variance_adapter_config)
        self.decoder = Decoder(**decoder_config)
        self.mel_linear = nn.Linear(decoder_config['decoder_hidden'], mel_size)
        self.postnet = PostNet(n_mel_channels=mel_size, **postnet_config)

        if self.is_gst:
            self.speaker_emb = GST()
        else:
            self.speaker_emb = nn.Embedding(n_speakers, encoder_config['encoder_hidden'])


    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        ref_mel=None
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            if self.is_gst:
                emb = self.speaker_emb(ref_mel)
                output = output + emb
            else:
                output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                    -1, max_src_len, -1
                )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )


@register_model_architecture('fast_speech2', 'fast_speech2_vctk')
def fast_speech2_vctk():
    return {
        'encoder_config': {
            'max_seq_len': 1000,
            'encoder_hidden': 256,
            'encoder_layer': 4,
            'encoder_head': 2,
            'conv_filter_size': 1024,
            'conv_kernel_size': [9, 1],
            'encoder_dropout': 0.2
        },
        'decoder_config': {
            'max_seq_len': 1000,
            'decoder_hidden': 256,
            'decoder_layer': 6,
            'decoder_head': 2,
            'conv_filter_size': 1024,
            'conv_kernel_size': [9, 1],
            'decoder_dropout': 0.2
        },
        'postnet_config': {
            'postnet_embedding_dim': 512,
            'postnet_kernel_size': 5,
            'postnet_n_convolutions': 5
        },
        'variance_adapter_config': {
            'pitch_feature': 'phoneme_level',
            'energy_feature': 'phoneme_level',
            'pitch_quantization': 'linear',
            'energy_quantization': 'linear',
            'n_variance_bins': 256,  # n_bins
            'stats': {
                'pitch': {
                    'min': -1.9287127187455897,
                    'max':  10.544469081998987
                },
                'energy': {
                    'min': -1.375638484954834,
                    'max': 8.256172180175781
                }
            },
            'encoder_hidden': 256,
            'filter_size': 256,
            'kernel_size': 3,
            'dropout': 0.5
        },
        'mel_size': 80,
        'n_speakers': 102,
        'is_gst': False
    }


@register_model_architecture('fast_speech2', 'fast_speech2_vctk_gst')
def fast_speech2_vctk_gst():
    d = fast_speech2_vctk()
    d['is_gst'] = True
    return d
