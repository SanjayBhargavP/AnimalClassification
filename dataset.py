import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_datasets(self):
        train_dataset = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=self.transform)
        val_dataset = datasets.ImageFolder(root=f"{self.data_dir}/val", transform=self.transform)
        test_dataset = datasets.ImageFolder(root=f"{self.data_dir}/test", transform=self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, len(train_dataset.classes)