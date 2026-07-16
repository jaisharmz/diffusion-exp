import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FlowMatcher
from inference import make_plot, plot_one

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 3
flow_matcher = FlowMatcher(28*28, 16).to(device)
autoencoder = flow_matcher.autoencoder
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
autoencoder_filename = "ae.pth"

if os.path.exists(autoencoder_filename):
    print("Loading Autoencoder...")
    autoencoder.load_state_dict(torch.load(autoencoder_filename, weights_only=True))
else:
    print("Training Autoencoder...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        running_train_loss = 0.0
        count_train_loss = 0
        running_test_loss = 0.0
        count_test_loss = 0

        autoencoder.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            loss = autoencoder.loss(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss = running_train_loss + loss.item()
            count_train_loss += 1
        avg_train_loss = running_train_loss / count_train_loss
        print(f"Train Loss: {avg_train_loss}")
        
        autoencoder.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                loss = autoencoder.loss(data)
                running_test_loss = running_test_loss + loss.item()
                count_test_loss += 1
                if batch_idx % 500 == 0:
                    x = data[0:1]
                    x_recon = autoencoder(x)
                    x, x_recon = x[0, 0].cpu().numpy(), x_recon[0, 0].cpu().numpy()
                    make_plot(x, x_recon, f"eval_epoch_{epoch}_{batch_idx}")
        avg_test_loss = running_test_loss / count_test_loss
        print(f"Test Loss: {avg_test_loss}")
        torch.save(autoencoder.state_dict(), autoencoder_filename)
    print("\n\n")

print("Training Flow Matcher...")
num_epochs = 50
for param in autoencoder.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, flow_matcher.parameters()), lr=0.001)
flow_matcher_filename = "fm.pth"

if os.path.exists(flow_matcher_filename):
    print("Loading Flow Matcher...")
    # set weights_only=False here if you hit a strict dict matching issue on the newly added embeddings
    flow_matcher.load_state_dict(torch.load(flow_matcher_filename, weights_only=False), strict=False)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} / {num_epochs}")
    running_train_loss = 0.0
    count_train_loss = 0
    running_test_loss = 0.0
    count_test_loss = 0

    flow_matcher.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # CFG Drop-out: 10% chance to set target to Null class (10)
        drop_mask = torch.rand(target.shape[0], device=device) < 0.1
        target[drop_mask] = 10 
        
        loss = flow_matcher.loss(data, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss = running_train_loss + loss.item()
        count_train_loss += 1
    avg_train_loss = running_train_loss / count_train_loss
    print(f"Train Loss: {avg_train_loss}")
    
    flow_matcher.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            loss = flow_matcher.loss(data, target)
            running_test_loss = running_test_loss + loss.item()
            count_test_loss += 1
            
            if batch_idx % 500 == 0:
                # Tell model to explicitly generate digits 0 through 9
                y_gen = torch.arange(10, device=device)
                
                x = flow_matcher.generate(y=y_gen, cfg_scale=3.0, num_steps=10)
                x = x.squeeze(1).cpu().numpy()
                
                for i in range(10):
                    plot_one(x[i], f"img_epoch_{epoch}_batch_{batch_idx}_digit_{i}")
                    
    avg_test_loss = running_test_loss / count_test_loss
    print(f"Test Loss: {avg_test_loss}")
    torch.save(flow_matcher.state_dict(), flow_matcher_filename)