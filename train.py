import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import os

class MultiLabelRepresentationProbe:
    def __init__(self, input_dim, num_labels):
        """
        input_dim: size of your activation vectors (4096)
        num_labels: number of binary labels to predict (e.g., 10)
        """
        self.model = nn.Linear(input_dim, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def train_probe(self, X_train, y_train, batch_size=4, epochs=20, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def evaluate_probe(self, X, y, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            outputs = torch.sigmoid(self.model(torch.FloatTensor(X)))
            predictions = (outputs > threshold).float().numpy()
            micro_f1 = f1_score(y, predictions, average='micro')
            macro_f1 = f1_score(y, predictions, average='macro')
            accuracies = [accuracy_score(y[:, i], predictions[:, i]) for i in range(y.shape[1])]
            return {
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'accuracies': accuracies
            }
    
def average_accuracies_for_the_same_predicate(self, accuracies:list, labels:list[str]) -> list:
    """average the accuracies across columns that have the same predicate. For example, `grasped cookies_1` and `grasped plate_1` have the same predicate
    Args:
        accuracies (list): a matrix of accuracies
        labels: labels of the columns of the matrix aka the symbols e.g. `grasped cookies_1` and `grasped plate_1`.
    Returns:
        accuracies averaged across columns with the same predicate
    """
    from collections import OrderedDict
    
    # Dictionary to store column indices by predicate in the order they appear
    predicate_to_indices = OrderedDict()
    
    # Extract predicates and group their corresponding column indices
    for i, label in enumerate(labels):
        predicate = label.split()[0]
        if predicate not in predicate_to_indices:
            predicate_to_indices[predicate] = []
        predicate_to_indices[predicate].append(i)
    
    # Now, compute the averaged accuracies for each row across the columns of the same predicate
    averaged_accuracies = []
    for row in accuracies:
        averaged_row = []
        for predicate, indices in predicate_to_indices.items():
            # Gather all values for this predicate
            vals = [row[idx] for idx in indices]
            # Compute the average
            avg_val = sum(vals) / len(vals) if vals else 0.0
            averaged_row.append(avg_val)
        averaged_accuracies.append(averaged_row)
    
    return averaged_accuracies


def load_probe_data(directory, exclude_files=None):
    """
    Load aggregated embeddings and labels from .pt files in the specified directory.

    Args:
        directory (str): Path to the directory containing .pt files.
        exclude_files (list, optional): List of filenames to exclude.

    Returns:
        tuple: (embeddings, labels) as NumPy arrays.
    """
    embeddings = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith(".pt") and file.startswith("episode_"):
            if exclude_files and file in exclude_files:
                continue
            data = torch.load(os.path.join(directory, file))
            embedding = data["visual_semantic_encoding"].numpy()  # Shape: (4096,)
            label = data["symbolic_state"].numpy()               # Shape: (num_labels,)
            embeddings.append(embedding)
            labels.append(label)
    return np.array(embeddings), np.array(labels)

if __name__ == "__main__":
    # Path to the directory containing aggregated .pt files
    data_dir = "openvla/experiments/logs"  # Update this path as needed

    # List of failed episodes to exclude
    exclude = ["episode_3.pt"]       # Add any other failed episodes here

    # Load aggregated data
    print("Loading aggregated data...")
    activations, labels = load_probe_data(data_dir, exclude_files=exclude)
    print(f"Loaded {activations.shape[0]} samples with activation dimension {activations.shape[1]} and {labels.shape[1]} labels.")

    # Verify shapes
    assert activations.shape[1] == 4096, f"Expected activation dimension 4096, got {activations.shape[1]}"
    # Ensure labels are binary
    assert set(np.unique(labels)) <= {0, 1}, "Labels should be binary (0 or 1)."

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Define probe parameters
    activation_dim = activations.shape[1]  # 4096
    num_labels = labels.shape[1]           # e.g., 10

    # Create and train probe
    probe = MultiLabelRepresentationProbe(
        input_dim=activation_dim,
        num_labels=num_labels
    )

    # Train
    print("Training probe...")
    probe.train_probe(X_train, y_train, epochs=20, batch_size=4)

    # Evaluate
    print("Evaluating probe on training data...")
    train_metrics = probe.evaluate_probe(X_train, y_train)
    print("Evaluating probe on testing data...")
    test_metrics = probe.evaluate_probe(X_test, y_test)

    # Display Metrics
    print(f"\nTraining Metrics:")
    print(f"Micro F1 Score: {train_metrics['micro_f1']:.3f}")
    print(f"Macro F1 Score: {train_metrics['macro_f1']:.3f}")
    print(f"Per-label Accuracies: {[f'{acc:.3f}' for acc in train_metrics['accuracies']]}")

    print(f"\nTesting Metrics:")
    print(f"Micro F1 Score: {test_metrics['micro_f1']:.3f}")
    print(f"Macro F1 Score: {test_metrics['macro_f1']:.3f}")
    print(f"Per-label Accuracies: {[f'{acc:.3f}' for acc in test_metrics['accuracies']]}")
