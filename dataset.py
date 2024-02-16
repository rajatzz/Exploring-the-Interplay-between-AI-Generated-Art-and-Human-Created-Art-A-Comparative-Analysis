from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, random_split

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

def load_dataset(root, batch_size=32):
    dataset = ImageFolder(root="Images", transform=transform, loader=lambda x: Image.open(x).convert("RGB"))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
