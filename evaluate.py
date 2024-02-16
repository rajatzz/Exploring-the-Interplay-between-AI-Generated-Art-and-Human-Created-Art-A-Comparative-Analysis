import torch
from torch.utils.data import DataLoader
from model import ArtClassifier
from dataset import load_dataset

def evaluate(model, test_loader):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for evaluation
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions and true labels for later analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

    # Calculate additional metrics (precision, recall, F1 score)
    confusion_matrix = torch.zeros(2, 2)
    for i in range(len(all_labels)):
        confusion_matrix[all_labels[i], all_predictions[i]] += 1

    true_positives = confusion_matrix[1, 1]
    false_positives = confusion_matrix[0, 1]
    false_negatives = confusion_matrix[1, 0]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

if __name__ == "__main__":
    # Load your dataset
    _, test_loader = load_dataset("Images")

    # Initialize the model
    model = ArtClassifier()

    # Load the trained model state
    model.load_state_dict(torch.load("Train/model_epoch_9.pth"))

    # Evaluate the model
    evaluate(model, test_loader)
