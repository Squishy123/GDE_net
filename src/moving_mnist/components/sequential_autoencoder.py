import torch
import torch.nn as nn

from collections import OrderedDict

'''
State Prediction Autoencoder Class
'''

class Sequential_Autoencoder(nn.Module):
    # accepts input for number of transforms
    def __init__(self, num_frames=1, dim_x=80, dim_y=80, channels=64):
        super(Sequential_Autoencoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(num_frames * channels, 16, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu1', nn.ReLU()),
            ('encoder_conv2',  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu2', nn.ReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_Tconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_Tconv3', nn.ConvTranspose2d(16, num_frames * channels, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu3', nn.ReLU()),
        ]))

    # forward pass
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x