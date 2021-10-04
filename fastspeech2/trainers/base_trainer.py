import torch
import torch.nn as nn
from typing import Tuple, Dict
from collections import defaultdict

from pytorch_sound.trainer import LogType, Trainer
from pytorch_sound.utils.tensor import to_device
from pytorch_sound.utils.calculate import db2log
from pytorch_sound.settings import MIN_DB
from fastspeech2.utils.tools import to_device, log
from fastspeech2.models.loss import FastSpeech2Loss
from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN


class BaseTrainer(Trainer):

    def __init__(self, model: nn.Module,
                 optimizer, train_dataset, valid_dataset,
                 max_step: int, valid_max_step: int, save_interval: int, log_interval: int,
                 pitch_feature: str, energy_feature: str,
                 save_dir: str, save_prefix: str = '',
                 grad_clip: float = 0.0, grad_norm: float = 0.0,
                 sr: int = 22050, pretrained_path: str = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 seed: int = 2021, is_reference: bool = False):
        super().__init__(model, optimizer, train_dataset, valid_dataset,
                         max_step, valid_max_step, save_interval, log_interval, save_dir, save_prefix,
                         grad_clip, grad_norm, pretrained_path, sr=sr, scheduler=scheduler, seed=seed)
        # vocoder
        self.interface = InterfaceHifiGAN(
            model_name='hifi_gan_v1_universal', device='cuda'
        )

        # make loss
        self.loss_func = FastSpeech2Loss(pitch_feature, energy_feature)

        self.is_reference = is_reference
        self.mel_log_min = db2log(MIN_DB)

    def forward(self, *inputs, is_logging: bool = False) -> Tuple[torch.Tensor, Dict]:
        # Forward
        output = self.model(*inputs[2:])

        # calculate loss
        losses = self.loss_func(inputs, output)
        loss = losses[0]  # total loss

        if is_logging:
            id_, text = inputs[:2]
            total_loss, mel_loss, post_loss, pitch_loss, energy_loss, duration_loss = losses

            raugh_mel, post_mel = output[:2]
            raugh_mel, post_mel = raugh_mel[:1].transpose(1, 2), post_mel[:1].transpose(1, 2)
            target_mel = inputs[6][:1].transpose(1, 2)

            # slice padded part, minimum value of mel is zero.
            if any([self.mel_log_min + 1 > numb for numb in target_mel[0, 0].cpu().numpy().tolist()]):
                first_pad_idx = int(target_mel[0, 0].argmin().cpu().numpy())
                raugh_mel = raugh_mel[..., :first_pad_idx]
                post_mel = post_mel[..., :first_pad_idx]
                target_mel = target_mel[..., :first_pad_idx]

            # synthesis
            pred_wav = self.interface.decode(post_mel).squeeze()
            rec_wav = self.interface.decode(target_mel).squeeze()
            raugh_mel, post_mel, target_mel = raugh_mel[0], post_mel[0], target_mel[0]

            meta = {
                # losses
                'total_loss': (total_loss.item(), LogType.SCALAR),
                'mel_loss': (mel_loss.item(), LogType.SCALAR),
                'post_loss': (post_loss.item(), LogType.SCALAR),
                'pitch_loss': (pitch_loss.item(), LogType.SCALAR),
                'energy_loss': (energy_loss.item(), LogType.SCALAR),
                'duration_loss': (duration_loss.item(), LogType.SCALAR),
                # plots
                'mel.target': (target_mel, LogType.IMAGE),
                'mel.raugh': (raugh_mel, LogType.IMAGE),
                'mel.post': (post_mel, LogType.IMAGE),
                'wav.target.plot': (rec_wav, LogType.PLOT),
                'wav.target.audio': (rec_wav, LogType.AUDIO),
                'wav.pred.plot': (pred_wav, LogType.PLOT),
                'wav.pred.audio': (pred_wav, LogType.AUDIO),
                # text
                'id_': (id_[0], LogType.TEXT),
                'text': (text[0], LogType.TEXT)
            }
        else:
            meta = {}
        return loss, meta

    @staticmethod
    def repeat(iterable):
        while True:
            for group in iterable:
                for x in group:
                    yield to_device(x, 'cuda')

    def train(self, step: int) -> torch.Tensor:

        # update model
        self.optimizer.zero_grad()

        # flag for logging
        log_flag = step % self.log_interval == 0

        # forward model
        loss, meta = self.forward(*next(self.train_dataset), is_logging=log_flag)

        # check loss nan
        if loss != loss:
            log('{} cur step NAN is occured'.format(step))
            return

        loss.backward()
        self.clip_grad()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # logging
        if log_flag:
            # console logging
            self.console_log('train', meta, step)
            try:
                # tensorboard logging
                self.tensorboard_log('train', meta, step)
            except OverflowError:
                pass

    def run(self):
        try:
            # training loop
            for i in range(self.step + 1, self.max_step + 1):

                # update step
                self.step = i

                # logging
                if i % self.save_interval == 1:
                    log('------------- TRAIN step : %d -------------' % i)

                # do training step
                self.model.train()
                self.train(i)

                # save model
                if i % self.save_interval == 0:
                    # save model checkpoint file
                    self.save(i)

        except KeyboardInterrupt:
            log('Train is canceled !!')