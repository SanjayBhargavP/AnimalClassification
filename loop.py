from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from visualize import Visualizer
from tabulate import tabulate


class Runner:
    def __init__(self, model, train_loader, val_loader, device, num_epochs, learning_rate):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.num_epochs = num_epochs

        # Lists to store metrics
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []

    def train(self):
        epoch_results = []

        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training phase
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}/{self.num_epochs}] Training"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / len(self.train_loader)
            epoch_train_accuracy = 100 * correct_train / total_train
            self.training_losses.append(epoch_train_loss)
            self.training_accuracies.append(epoch_train_accuracy)

            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in tqdm(self.val_loader, desc=f"Epoch [{epoch + 1}/{self.num_epochs}] Validation"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = val_loss / len(self.val_loader)
            epoch_val_accuracy = 100 * correct_val / total_val
            self.validation_losses.append(epoch_val_loss)
            self.validation_accuracies.append(epoch_val_accuracy)

            # Store epoch results for tabulation
            epoch_results.append([
                epoch + 1,
                f"{epoch_train_loss:.4f}",
                f"{epoch_train_accuracy:.2f}%",
                f"{epoch_val_loss:.4f}",
                f"{epoch_val_accuracy:.2f}%"
            ])

        # Print tabulated epoch results
        headers = ["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
        print("\nTraining Summary:")
        print(tabulate(epoch_results, headers=headers, tablefmt="grid"))

        # Visualize the training and validation losses and accuracies
        Visualizer.plot_losses(self.training_losses, self.validation_losses)
        Visualizer.plot_accuracies(self.training_accuracies, self.validation_accuracies)

    def test(self, test_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing Progress"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # Calculate overall test accuracy
        test_accuracy = 100 * correct_test / total_test

        # Print test accuracy
        print("\nTest Summary:")
        headers = ["Test Accuracy"]
        results = [[f"{test_accuracy:.2f}%"]]
        print(tabulate(results, headers=headers, tablefmt="grid"))

        return all_labels, all_preds
