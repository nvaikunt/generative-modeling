import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        input_channels, height, width, = input_shape
        """
        TODO 2.1 : Fill in self.convs following the given architecture
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2)
        )

        #TODO 2.1: fill in self.fc, such that output dimension is self.latent_dim
        self.fc = nn.Linear(int(256 * (height / 8) * (width / 8)), self.latent_dim)

    def forward(self, x):
        #TODO 2.1 : forward pass through the network, output should be of dimension : self.latent_dim
        features = self.convs(x)
        flattened_features = features.reshape(x.size(dim=0), -1)
        return self.fc(flattened_features)

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        in_channels, height, width = input_shape
        #TODO 2.4: fill in self.fc, such that output dimension is 2*self.latent_dim
        self.fc = nn.Linear(int(256 * (height / 8) * (width / 8)), 2 * self.latent_dim)

    def forward(self, x):
        #TODO 2.4: forward pass through the network.
        # should return a tuple of 2 tensors, mu and log_std
        features = self.convs(x)
        flattened_features = features.reshape(x.size(dim=0), -1)
        lin_out = self.fc(flattened_features)
        mu, log_std = torch.split(lin_out, 
                                  split_size_or_sections=self.latent_dim, dim=-1)
        return mu, log_std


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        output_channels, height, width = output_shape

        #TODO 2.1: fill in self.base_size
        self.base_size = (256, int(height / 8), int(width / 8))
        self.fc = nn.Linear(self.latent_dim, int(256 * (height / 8) * (width / 8)))

        """
        TODO 2.1 : Fill in self.deconvs following the given architecture
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.deconvs = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), 
            nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            nn.ReLU(), nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, z):
        #TODO 2.1: forward pass through the network, first through self.fc, then self.deconvs
        lin_out = self.fc(z)
        base_imgs = lin_out.reshape(z.size(dim=0), *self.base_size)
        out = self.deconvs(base_imgs)
        return out

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    #NOTE: You don't need to implement a forward function for AEModel. For implementing the loss functions in train.py, call model.encoder and model.decoder directly.
