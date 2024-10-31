from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn

class Runner:
    def __init__(self, model, train_loader, val_loader, device, num_epochs, learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.num_epochs = num_epochs

    
    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.num_epochs}] Training"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward + Backward + Optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(self.train_loader):.4f}")
            
            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(self.val_loader, desc=f"Epoch [{epoch+1}/{self.num_epochs}] Validation"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total
            print(f"Validation Loss: {val_loss/len(self.val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    def test(self, test_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing Progress"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        return all_labels, all_preds