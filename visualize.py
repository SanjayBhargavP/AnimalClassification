import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_losses(training_losses, validation_losses):
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_accuracies(training_accuracies, validation_accuracies):
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(training_accuracies) + 1), training_accuracies, label='Training Accuracy')
        plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid()
        plt.show()