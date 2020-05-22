import torch
import numpy as np
import torch.nn as nn

class Reshape(nn.Module):
    def forward(self, input, shape=[-1,3,32,32]):
        # resize the data as (BATCH_SIZE,3,32,32)
        return torch.reshape(input,shape)

# flatten a tensor to one column vector
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)

class Reconstruction(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class ConstructImg(nn.Module):
    def forward(self, input, shape=[-1,32,32,3]):
        # resize the data as (BATCH_SIZE,3,32,32)
        return torch.reshape(input,shape)

class VAE(nn.Module):
    def __init__(self, amt_channel = 3, keep_prob=0.6,enc_dim= 1024, z_dim=64):
        super(VAE,self).__init__()
        self.encoder = nn.Sequential(
            Reshape(),
            nn.Conv2d(amt_channel, 32, kernel_size=4, stride=2, padding=1), # output shape = (BATCH_SIZE,32,16,16)
            nn.ELU(alpha=0.2),
            nn.Dropout(keep_prob),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # output shape = (BATCH_SIZE,64,8,8)
            nn.ELU(alpha=0.2),
            nn.Dropout(keep_prob),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# output shape = (BATCH_SIZE,128,4,4)
            nn.ELU(alpha=0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# output shape = (BATCH_SIZE,256,2,2)
                                                                    # enc_dim = 256*2*2
            nn.ELU(alpha=0.2),
            nn.Dropout(keep_prob),
            Flatten()
        )
        # enc2latent is used for computing latent distribution
        self.enc2latent_mu = nn.Linear(enc_dim, z_dim)
        # self.enc2latent_var = nn.Linear(enc_dim, z_dim)
        #latent2dec is used for convert latent variable into decoder
        self.latent2dec = nn.Linear(z_dim, enc_dim)

        self.decoder = nn.Sequential(
            Reconstruction(),                                           # output shape = (BATCH_SIZE,1024,1,1)
            nn.ConvTranspose2d(enc_dim, 128, kernel_size=3, stride=2),  # output shape = (BATCH_SIZE,128,3,3)
            nn.ReLU(),
            nn.Dropout(keep_prob),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),       # output shape = (BATCH_SIZE,64,7,7)
            nn.ReLU(),
            nn.Dropout(keep_prob),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),# output shape = (BATCH_SIZE,32,15,15)
            nn.ReLU(),
            nn.ConvTranspose2d(32, amt_channel, kernel_size=4, stride=2),# output shape = (BATCH_SIZE,3,32,32)
            nn.Sigmoid(),
            ConstructImg()
        )

    def generate_z(self,x):
        latent_x = self.enc2latent_mu(x)
        latent_noise = torch.randn(*latent_x.size())
        latent_logvar = latent_x.mul(0.5).exp_()
        z = latent_x + latent_logvar * latent_noise
        # z = latent_x + torch.mm(latent_noise,latent_logvar)
        return z,latent_x

    def forward(self,x):
        x = self.encoder(x)
        z, latent_x = self.generate_z(x)
        dec = self.latent2dec(z)
        img = self.decoder(dec)
        return img, latent_x
