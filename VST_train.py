import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from VST_custom_dataset import VSTDataset
from torchvision import transforms
from tqdm import tqdm
import argparse
from VST_GAN import Generator, Discriminator
from pytorch_fid import fid_score 
import matplotlib.pyplot as plt
from VST_textProcessor import TextProcessor

def train_vst(generator, discriminator, optimizerD, optimizerG, criterion, train_dataloader, val_dataloader, device, num_epochs, save_path, batch_size, text_processor):
    
    writer = SummaryWriter('vst_gan_logs')
    save_interval = 10
    validation_interval = 1

    discriminator_losses = []
    generator_losses = []

    for epoch in tqdm(range(num_epochs)):
        discriminator.train()
        generator.train()

        for i, data in enumerate(train_dataloader, 0):
            # Unpack data
            input_ids, attention_mask, real_data = data
            input_ids, attention_mask, real_data = input_ids.to(device), attention_mask.to(device), real_data.to(device)

            # Generate text embeddings
            text_embeddings = text_processor.get_embeddings(input_ids, attention_mask)

            # Train Discriminator with real data
            discriminator.zero_grad()
            real_labels = torch.full((real_data.size(0),), 1, dtype=torch.float, device=device)
            
            # Forward pass through discriminator (with text embeddings)
            output = discriminator(real_data, text_embeddings).view(-1)
            errD_real = criterion(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()

            # Train Discriminator with fake data
            noise = torch.randn(real_data.size(0), 100, 1, 1, device=device)
            fake_data = generator(noise, text_embeddings)
            fake_labels = torch.full((real_data.size(0),), 0, dtype=torch.float, device=device)
            
            output = discriminator(fake_data.detach(), text_embeddings).view(-1)
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_data, text_embeddings).view(-1)
            errG = criterion(output, real_labels)  # Note: real_labels used here
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            discriminator_losses.append(errD.item())
            generator_losses.append(errG.item())

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(train_dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
                writer.add_scalar('Loss/Discriminator', errD.item(), len(train_dataloader)*epoch + i)
                writer.add_scalar('Loss/Generator', errG.item(), len(train_dataloader)*epoch + i)

        # Validation
        if epoch % validation_interval == 0:
            generator.eval()  # Set generator to evaluation mode
            discriminator.eval()  # Set discriminator to evaluation mode
            val_loss_d = 0.0
            val_loss_g = 0.0
            with torch.no_grad():
                for val_data in val_dataloader:
                    # Unpack data and generate text embeddings
                    input_ids, attention_mask, val_real_data = val_data
                    input_ids, attention_mask, val_real_data = input_ids.to(device), attention_mask.to(device), val_real_data.to(device)
                    val_text_embeddings = text_processor.get_embeddings(input_ids, attention_mask)

                    val_batch_size = val_real_data.size(0)
                    val_real_labels = torch.full((val_batch_size,), 1, dtype=torch.float, device=device)

                    # Validation Discriminator loss
                    val_output_d = discriminator(val_real_data, val_text_embeddings).view(-1)
                    val_errD_real = criterion(val_output_d, val_real_labels)
                    val_loss_d += val_errD_real.item()

                    # Validation Generator loss
                    val_noise = torch.randn(val_batch_size, 100, 1, 1, device=device)
                    val_fake_data = generator(val_noise, val_text_embeddings)
                    val_fake_labels = torch.full((val_batch_size,), 0, dtype=torch.float, device=device)
                    val_output_g = discriminator(val_fake_data, val_text_embeddings).view(-1)
                    val_errG = criterion(val_output_g, val_fake_labels)
                    val_loss_g += val_errG.item()

                val_loss_d /= len(val_dataloader)
                val_loss_g /= len(val_dataloader)

            print(f'Validation Loss_D: {val_loss_d:.4f} Validation Loss_G: {val_loss_g:.4f}')
            writer.add_scalar('Validation Loss/Discriminator', val_loss_d, epoch)
            writer.add_scalar('Validation Loss/Generator', val_loss_g, epoch)
            generator.train()  # Set generator back to training mode
            discriminator.train()  # Set discriminator back to training mode

        if epoch % save_interval == 0:
            torch.save(generator.state_dict(), os.path.join('checkpoints', f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join('checkpoints', f'discriminator_epoch_{epoch}.pth'))
    
    writer.close()
    torch.save(generator.state_dict(), save_path)
    
    # Create and save the loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(discriminator_losses)), discriminator_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(range(len(generator_losses)), generator_losses, label='Generator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Losses Over Training')
    plt.savefig('gan_losses.png')

    # Optionally, display the loss plot
    plt.show()

    print('Training finished.')

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('-tl', '--training_loss', type=str, default='training_loss.png', help='Name of the training loss to be saved')
    parser.add_argument('-s', '--save', type=str, default='yoda.pth', help='Name to save the model under')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epoch = args.epochs
    batch = args.batch_size
    learning_rate = args.lr
    model_loc = args.save

    # Data loading and transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize tokenizer (BERT tokenizer in this example)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_processor = TextProcessor()

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

    train_vst(netG, netD, optimizerD, optimizerG, criterion, train_loader, val_loader,device, num_epoch, model_loc, batch, text_processor)
