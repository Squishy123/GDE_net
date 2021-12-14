import torch
import torch.nn as nn

from collections import OrderedDict

'''
State Prediction Autoencoder Class
'''

class Sequential_Autoencoder(nn.Module):
    # accepts input for number of transforms
    def __init__(self, frame_stacks=1, channels=32, num_frames=1):
        super(Sequential_Autoencoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(channels * frame_stacks * num_frames, 16, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu1', nn.ReLU()),
            ('encoder_conv2',  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu2', nn.ReLU()),
            ('encoder_conv3', nn.Conv2d(32, 32, kernel_size=7)),
            ('encoder_relu3', nn.ReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_Tconv1', nn.ConvTranspose2d(32, 32, kernel_size=7)),
            ('decoder_relu1', nn.ReLU()),
            ('decoder_Tconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_Tconv3', nn.ConvTranspose2d(16, channels * frame_stacks * num_frames, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu3', nn.ReLU()),
        ]))

    # forward pass
    def forward(self, x):
        x = self.encoder(x)
        x1 = self.decoder(x)
        return x1