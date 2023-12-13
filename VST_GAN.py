import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, text_embedding_size=768, ngf=64, nc=3):
        super(Generator, self).__init__()
        # Linear layer to transform the text embeddings
        self.embedding_transform = nn.Linear(text_embedding_size, nz)

        # Adjust the generator architecture
        self.main = nn.Sequential(
            # First layer takes the concatenated noise and transformed text embeddings
            nn.ConvTranspose2d(nz * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Subsequent layers
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Final layer to produce the output image
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, text_embeddings):
        # Transform the text embeddings
        transformed_embeddings = self.embedding_transform(text_embeddings)
        transformed_embeddings = transformed_embeddings.view(text_embeddings.size(0), -1, 1, 1)

        # Concatenate noise and transformed text embeddings
        combined_input = torch.cat([noise, transformed_embeddings], dim=1)

        # Pass the combined input through the generator network
        return self.main(combined_input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, text_embedding_size=768, output_size=(7, 7)):
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
        self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size)
        # Adjust the size accordingly based on your input image dimensions
        self.flatten = nn.Flatten(start_dim=1)
        
        # Calculate the input size for the linear layers
        # image_feature_size = (ndf * 8 * 14 * 14)  # Update this based on your image size and architecture
        adaptive_output_size = output_size[0] * output_size[1] * ndf * 8
        linear_input_size = adaptive_output_size + text_embedding_size

        # Additional dense layers after concatenating with text embeddings
        self.dense_layers = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embeddings):
        # Process image through conv layers
        image_features = self.main(image)
        
        # Flatten image_features
        image_features = self.adaptive_pool(image_features)
        
        image_features = self.flatten(image_features)

        # Concatenate image features with text embeddings
        combined_input = torch.cat([image_features, text_embeddings.view(text_embeddings.size(0), -1)], dim=1)

        # Forward through dense layers
        return self.dense_layers(combined_input)
