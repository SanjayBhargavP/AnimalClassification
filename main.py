import json
import torch
import torch.optim as optim
from dataset import DatasetLoader
from model import *
from loop import Runner
from metrics import Metrics

# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

# Define hyperparameters
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]
DATA_DIR = config["data_dir"]
MODEL_NAME = config["model_name"]

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
dataset_loader = DatasetLoader(DATA_DIR, BATCH_SIZE)
train_loader, val_loader, test_loader, num_classes = dataset_loader.load_datasets()

# Load model dynamically based on config file
model_class = globals()[MODEL_NAME]
model = model_class(num_classes)

# Train and validate model
runner = Runner(model, train_loader, val_loader, device, NUM_EPOCHS, LEARNING_RATE)
runner.train()

# Test model
all_labels, all_preds = runner.test(test_loader)

# Compute and print metrics
precision, recall, f1 = Metrics.compute_metrics(all_labels, all_preds)
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")