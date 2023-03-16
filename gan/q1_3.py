import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    bce_real = torch.nn.BCEWithLogitsLoss()
    bce_fake = torch.nn.BCEWithLogitsLoss()
    targets_real = torch.ones_like(discrim_real).cuda()
    targets_fake = torch.zeros_like(discrim_fake).cuda()
    loss_real = bce_real(discrim_real, targets_real)
    loss_fake = bce_fake(discrim_fake, targets_fake)
    loss = loss_real + loss_fake
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """
    bce = torch.nn.BCEWithLogitsLoss()
    targets = torch.ones_like(discrim_fake).cuda()
    loss = bce(discrim_fake, targets)
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
