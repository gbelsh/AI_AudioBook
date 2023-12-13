
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from VST_custom_dataset import VSTDataset
from torchvision import transforms
from tqdm import tqdm
import argparse
from VST_GAN import Generator, Discriminator
from torchvision.utils import save_image
from pytorch_fid import fid_score  # You might need to install this package


def train_vst(generator, discriminator, optimizerD, optimizerG, criterion, dataloader, device, num_epochs):

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Train Discriminator with real data
            discriminator.zero_grad()
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            real_labels = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            
            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()

            # Train Discriminator with fake data
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = generator(noise)
            fake_labels = torch.full((batch_size,), 0, dtype=torch.float, device=device)
            
            output = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            generator.zero_grad()
            fake_labels = torch.full((batch_size,), 1, dtype=torch.float, device=device)  # Labels are flipped for generator
            output = discriminator(fake_data).view(-1)
            errG = criterion(output, fake_labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

    print('Training finished.')



def eval_gan(generator, dataloader, device, fixed_noise, fid_real_path, epoch):
    # Switch model to evaluation mode
    generator.eval()

    # Generate and save images using the fixed noise
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    save_image(fake_images, f'output_epoch_{epoch}.png')

    # For FID score, you need to generate a large number of images and save them
    # Then use these generated images to calculate the FID score against real images
    # FID Score calculation (pseudo-code)
    # fid_value = fid_score.calculate_fid_given_paths([fid_real_path, fid_generated_path], batch_size, device, dims)

    # Switch model back to training mode
    generator.train()

    # Return evaluation metrics if necessary
    # return fid_value


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default=10, help='Train, Test, or Val')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('-tl', '--training_loss', type=str, default='training_loss.png', help='Name of the training loss to be saved')
    parser.add_argument('-s', '--save', type=str, default='yoda.pth', help='Name to save the model under')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ttv = args.type
    num_epoch = args.epochs
    batch = args.batch_size
    learning_rate = args.lr

    # Data loading and transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize tokenizer (BERT tokenizer in this example)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = VSTDataset('Datasets/train', tokenizer)
    val_dataset = VSTDataset('Datasets/validation', tokenizer)
    test_dataset = VSTDataset('Datasets/test', tokenizer)



    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Create the generator and discriminator
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    if (ttv == "train"):
        train_vst(netG, netD, optimizerD, optimizerG, criterion, train_loader, device, num_epoch)
    # elif (ttv == "eval"):
    #     eval_gan(netG, dataloader, device, fixed_noise, fid_real_path, epoch)
    # else:
    #     printf("Error type not accepted")
    #     exit()
        


   
