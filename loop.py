from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from metrics import Metrics
from datetime import datetime

class Runner:
    def __init__(self, model, train_loader, val_loader, test_loader, device, num_epochs, learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.num_epochs = num_epochs

    
    def train(self):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(f"training_metrics_per_epoch_{timestamp}.csv", "w") as f:
            f.write("Epoch,Validation Loss,Validation Accuracy,Precision,Recall,F1 Score\n")

        avg_val_loss = 0
        val_accuracy = 0
        precision = 0
        recall = 0
        f1 = 0

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
            
            avg_val_loss = running_loss/len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_val_loss:.4f}")
            
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

            all_labels, all_preds = self.test(self.test_loader)
            precision, recall, f1 = Metrics.compute_metrics(all_labels, all_preds)

            with open(f"training_metrics_per_epoch_{timestamp}.csv", "a") as f:
                f.write(f"{epoch+1},{avg_val_loss},{val_accuracy},{precision},{recall},{f1}\n")

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