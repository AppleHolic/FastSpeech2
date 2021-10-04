import fire
import torch
import torch.nn as nn
import json
from typing import Tuple
from pytorch_sound.models import build_model
from torch.utils.data import DataLoader
from fastspeech2.trainers.base_trainer import BaseTrainer
from fastspeech2.dataset import Dataset


def main(train_path: str, preprocessed_path: str,
         save_dir: str, save_prefix: str,
         model_name: str, pretrained_path: str = '', num_workers: int = 16,
         batch_size: int = 16,
         pitch_feature: str = 'phoneme', energy_feature: str = 'phoneme',
         pitch_min: float = 0., energy_min: float = 0.,
         lr: float = 2e-4, weight_decay: float = 0.0001, betas=(0.9, 0.98),
         max_step: int = 200000, group_size: int = 4,
         save_interval: int = 10000, log_interval: int = 50, grad_clip: float = 0.0, grad_norm: float = 5.0,
         milestones: Tuple[int] = None, gamma: float = 0.2, sr: int = 22050, seed: int = 2021,
         is_reference: bool = False):
    # create model
    model = build_model(model_name).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # create optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    else:
        scheduler = None

    dataset = Dataset(train_path, preprocessed_path, pitch_min=pitch_min, energy_min=energy_min,
                      text_cleaners=['english_cleaners'],
                      batch_size=batch_size, sort=True, drop_last=True, is_reference=is_reference)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers
    )

    # train
    BaseTrainer(
        model, optimizer,
        train_loader, None,
        max_step=max_step, valid_max_step=1, save_interval=save_interval,
        log_interval=log_interval, pitch_feature=pitch_feature, energy_feature=energy_feature,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=pretrained_path, sr=sr,
        scheduler=scheduler, seed=seed, is_reference=is_reference
    ).run()


def run_config(config_path: str):
    with open(config_path, 'r') as r:
        config = json.load(r)
    main(**config)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    fire.Fire(run_config)
