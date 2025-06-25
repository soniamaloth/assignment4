import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 28 * 28  # MNIST images are 28x28
num_epochs = 20  # Reduced for faster training
batch_size = 128  # Increased for efficiency
lr = 0.0002
beta1 = 0.5

# MNIST Dataset (using a subset for speed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_dataset = torch.utils.data.Subset(train_dataset, range(10000))  # Use first 10,000 samples
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
print(f"Dataset size: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)  # Reshape to image format
        return img

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
print("Models initialized on device.")

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to track losses
d_losses = []
g_losses = []

# Function to save generated images
def save_generated_images(epoch, generator, fixed_noise, folder='output'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    filepath = f'{folder}/epoch_{epoch}.png'
    plt.savefig(filepath)
    plt.close()
    print(f"Image saved successfully at {filepath}")

# Fixed noise for consistent visualization
fixed_noise = torch.randn(64, latent_dim).to(device)

# Training Loop
try:
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}/{num_epochs}")
        for i, (imgs, _) in enumerate(train_loader):
            batch_size = imgs.size(0)
            
            # Ground truth labels
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Train on real data
            real_imgs = imgs.to(device)
            real_validity = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_validity, real_label)
            
            # Train on fake data
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_validity, fake_label)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generator wants Discriminator to think fake images are real
            fake_validity = discriminator(fake_imgs)
            g_loss = adversarial_loss(fake_validity, real_label)
            g_loss.backward()
            optimizer_G.step()
            
            # Save losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            
            if i % 10 == 0:  # Increased print frequency for better monitoring
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        # Save generated images at epochs 0, 10, 19
        if epoch in [0, 10, 19]:
            save_generated_images(epoch, generator, fixed_noise)
        print(f"Completed epoch {epoch} successfully.")

except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr)
    sys.exit(1)

# Plot Generator and Discriminator Losses
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('Generator and Discriminator Losses Over Time')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('output/loss_plot.png')
plt.close()
print("Training completed. Loss plot saved to output/loss_plot.png")