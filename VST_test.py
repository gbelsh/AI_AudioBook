import torch
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForImageGeneration
from torchvision.utils import save_image
import os

# Define the descriptions you want to compare (fill these in)
dalle_descriptions = ["Description 1 for DALL·E", "Description 2 for DALL·E", ...]
gan_descriptions = ["Description 1 for Custom GAN", "Description 2 for Custom GAN", ...]

# Initialize DALL·E model and feature extractor
dalle_model_name = "openai/clip-vit-base-patch16"
dalle_image_model_name = "openai/DALL·E-base-12-Image"

dalle_feature_extractor = AutoFeatureExtractor.from_pretrained(dalle_model_name)
dalle_model = AutoModelForImageGeneration.from_pretrained(dalle_image_model_name)

# Initialize your custom GAN model (fill this in)

# Directory to save generated images
output_dir = "comparison_images"
os.makedirs(output_dir, exist_ok=True)

# Loop through descriptions and generate images
for i, (dalle_desc, gan_desc) in enumerate(zip(dalle_descriptions, gan_descriptions)):
    # Generate image with DALL·E
    dalle_inputs = dalle_feature_extractor(dalle_desc, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    with torch.no_grad():
        dalle_output = dalle_model.generate(**dalle_inputs)
    save_image(dalle_output, os.path.join(output_dir, f"dalle_image_{i}.png"))

    # Generate image with custom GAN (fill this in)

# Visualize the generated images
for i in range(len(dalle_descriptions)):
    dalle_img = plt.imread(os.path.join(output_dir, f"dalle_image_{i}.png"))
    gan_img = plt.imread(os.path.join(output_dir, f"gan_image_{i}.png"))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(dalle_img)
    plt.title("DALL·E Generated")

    plt.subplot(1, 2, 2)
    plt.imshow(gan_img)
    plt.title("Custom GAN Generated")

    plt.tight_layout()
    plt.show()
