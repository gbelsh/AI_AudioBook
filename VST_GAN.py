import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, text_embedding_size=768, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz + text_embedding_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Additional layers
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, text_embeddings):
        # Concatenate noise and text embeddings
        combined_input = torch.cat([noise, text_embeddings], dim=1)
        return self.main(combined_input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, text_embedding_size=768):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is the image (3x64x64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()

        # Additional dense layers after concatenating with text embeddings
        self.dense_layers = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4 + text_embedding_size, 512),  # Adjust the size accordingly
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embeddings):
        # Process image through conv layers
        image_features = self.main(image)
        image_features = self.flatten(image_features)

        # Concatenate image features with text embeddings
        combined_input = torch.cat([image_features, text_embeddings], dim=1)

        # Forward through dense layers
        return self.dense_layers(combined_input)
