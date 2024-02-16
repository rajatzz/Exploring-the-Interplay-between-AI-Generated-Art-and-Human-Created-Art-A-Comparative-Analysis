import torch
from torch import nn, optim
from model import ArtClassifier
from dataset import load_dataset

# Initialize the model
model = ArtClassifier()

# Load your dataset
train_loader, test_loader = load_dataset("Images")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy * 100:.2f}%")

    # Save the model after each epoch
    torch.save(model.state_dict(), f"Train/model_epoch_{epoch + 1}.pth")
