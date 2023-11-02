from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from support import Generator, Discriminator
from PIL import Image
import torchvision.utils as vutils


real_label_value = 0.9
fake_label_value = 0.1

# Initialize
generator = Generator()
discriminator = Discriminator()

# Hyperparameters
batch_size = 128
lr = 0.0001

# Custom DataLoader
transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# Replace this with your DataLoader
train_data = datasets.ImageFolder(root="./images", transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# Optimizers and Loss
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)


# Learning Rate Scheduler
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.95)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.95)

# Training Loop
for epoch in range(2501):

    for batch_idx, (real_data, _) in enumerate(train_loader):
        batch_size = real_data.size(0)

        # Real and Fake labels with label smoothing
        real_label = torch.full((batch_size, 1), real_label_value)
        fake_label = torch.full((batch_size, 1), fake_label_value)

        # Train Discriminator
        optimizer_d.zero_grad()
        output = discriminator(real_data)
        loss_real = criterion(output, real_label)

        noise = torch.randn(batch_size, 100)
        fake_data = generator(noise).detach()
        output = discriminator(fake_data)
        loss_fake = criterion(output, fake_label)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, 100)
        fake_data = generator(noise)
        output = discriminator(fake_data)
        loss_g = criterion(output, real_label)

        loss_g.backward()
        optimizer_g.step()
        noise = torch.randn(1, 100)  # Generate one example with shape (1, 100)
        with torch.no_grad():  # No need to track gradients here
            fake_data = generator(noise)
            if epoch % 500 == 0:
                vutils.save_image(fake_data.view(fake_data.size(0), 3, 64, 64),
                          f'./generated_images/fake_image_epoch1_{epoch}.png',
                          normalize=True)

        scheduler_g.step()
        scheduler_d.step()
    print(f"Epoch {epoch} completed")

