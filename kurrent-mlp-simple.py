# A SIMPLE MLP USING TORCH
#============================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#============================================================================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.model(x)

def train(model, train_loader, criterion, optimizer, n_epochs):
    model.train()
    epochwise_losses = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            #fwd pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch: [{epoch+1}/{n_epochs}], loss: {running_loss/len(train_loader):.4f}")
        epochwise_losses.append(running_loss/len(train_loader))
    return epochwise_losses

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"accuracy: {accuracy:.2f}%")

#------------------------------------------------------
device = 'cpu'
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 10
losses = train(model, train_loader, criterion, optimizer, n_epochs)
evaluate(model, test_loader)
plt.figure(figsize=(8, 6))
plt.plot(range(1, n_epochs + 1), losses, marker='o', label='loss')
plt.title('loss x epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.show()