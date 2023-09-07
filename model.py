import torch
import torch.nn as nn
import numpy as np


def all_descending(*params):
    sorted_params = sorted(params, reverse=True)

    return np.array_equal(params, sorted_params)


def all_ascending(*params):
    sorted_params = sorted(params)

    return np.array_equal(params, sorted_params)


def create_modules(*params):
    assert len(params) >= 2

    layers = nn.ModuleList()

    for i, curr in enumerate(params[1:]):
        prev = params[i]

        # last layer does not have relu activation
        if i == len(params[1:]) - 1:
            layers.append(nn.Linear(prev, curr))
        else:
            layers.append(nn.Sequential(nn.Linear(prev, curr), nn.ReLU()))

    return layers


class Encoder(nn.Module):
    def __init__(self, *params):
        """
        Encoder module for the autoencoder.

        Parameters:
        - params: number of neurons in each layer.

        """
        super().__init__()

        # make sure params are in descending order
        assert all_descending(*params)

        self.layers = create_modules(*params)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, *params):
        """
        Decoder module for the autoencoder.

        Parameters:
        - params: number of neurons in each layer.

        """
        super().__init__()

        # make sure params are in ascending order
        assert all_ascending(*params)

        self.layers = create_modules(*params)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class AutoEncoder(nn.Module):
    def __init__(self, encoder_params: list[int], decoder_params: list[int]):
        super().__init__()

        # input to encoder is equal to output of decoder
        assert encoder_params[0] == decoder_params[-1]

        # output of encoder is equal to input to decoder
        assert encoder_params[-1] == decoder_params[0]

        self.encoder = Encoder(*encoder_params)
        self.decoder = Decoder(*decoder_params)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
