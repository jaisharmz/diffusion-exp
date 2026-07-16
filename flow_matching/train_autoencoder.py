import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Autoencoder
from inference import make_plot

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

device = "cuda"
num_epochs = 10
model = Autoencoder(28*28, 16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} / {num_epochs}")
    running_train_loss = 0.0
    count_train_loss = 0
    running_test_loss = 0.0
    count_test_loss = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = model.loss(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss = running_train_loss + loss
        count_train_loss += 1
    avg_train_loss = running_train_loss / count_train_loss
    print(f"Train Loss: {avg_train_loss}")
    
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        loss = model.loss(data)
        running_test_loss = running_test_loss + loss
        count_test_loss += 1
        if batch_idx % 500 == 0:
            x = data[0]
            x_recon = model(x)
            x, x_recon = x[0].cpu().detach().numpy(), x_recon[0].cpu().detach().numpy()
            make_plot(x, x_recon, f"eval_epoch_{epoch}_{batch_idx}")
    avg_test_loss = running_test_loss / count_test_loss
    print(f"Test Loss: {avg_test_loss}")
