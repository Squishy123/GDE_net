import torch
import torch.nn as nn

from collections import OrderedDict

'''
State Prediction Autoencoder Class
'''

class Sequential_Autoencoder(nn.Module):
    # accepts input for number of transforms
    def __init__(self, num_frames=20, dim_x=10, dim_y=10, channels=32):
        super(Sequential_Autoencoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(num_frames, 16, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu1', nn.ReLU()),
            ('encoder_conv2',  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu2', nn.ReLU()),
            ('encoder_conv3', nn.Conv2d(32, 32, kernel_size=1)),
            ('encoder_relu3', nn.ReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_Tconv1', nn.ConvTranspose2d(32, 32, kernel_size=1)),
            ('decoder_relu1', nn.ReLU()),
            ('decoder_Tconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_Tconv3', nn.ConvTranspose2d(16, num_frames, kernel_size=3, stride=2, padding=1)),
            ('decoder_relu3', nn.ReLU()),
        ]))

    # forward pass
    def forward(self, x):
        x = torch.flatten(x, 2).unsqueeze(3)
        #x = torch.reshape(x, (x.shape[0],x.shape[1],int((x.shape[2]*x.shape[3]*x.shape[4])**(0.5)+1),int((x.shape[2]*x.shape[3]*x.shape[4])**(0.5)-1)))
        x = self.encoder(x)
        x = self.decoder(x)
        return x