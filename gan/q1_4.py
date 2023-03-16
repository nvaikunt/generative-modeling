import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.4.1: Implement LSGAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    mse_real = torch.nn.MSELoss()
    mse_fake = torch.nn.MSELoss()

    target_real = torch.ones_like(discrim_real).cuda()
    target_fake = torch.zeros_like(discrim_fake).cuda()

    loss_real = mse_real(discrim_real, target_real)
    loss_fake = mse_fake(discrim_fake, target_fake)
    loss = .5 * loss_real + .5 * loss_fake
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.4.1: Implement LSGAN loss for generator.
    """
    mse = torch.nn.MSELoss()
    targets = torch.ones_like(discrim_fake).cuda()
    loss = mse(discrim_fake, targets)
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.4.2: Run this line of code.
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
