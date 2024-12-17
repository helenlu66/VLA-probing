import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import ast
import warnings

# --- Suppress FutureWarnings from torch.load ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Data Loading Function ---
def load_probe_data(directory, layer_idx, exclude_files=None, detectors=None):
    """
    Load per-action embeddings from a specific layer and separate labels for each detector.

    Args:
        directory (str): Path to the directory containing .pt files.
        layer_idx (int): The layer index to extract embeddings from.
        exclude_files (list, optional): List of filenames to exclude.
        detectors (list, optional): List of detector keys to extract labels for.

    Returns:
        tuple: (embeddings, labels_dict) where:
            - embeddings: NumPy array of shape (total_actions, hidden_dim)
            - labels_dict: Dictionary with keys as detector names and values as label arrays.
    """
    if detectors is None:
        detectors = ['symbolic_state_object_relations', 'symbolic_state_action_subgoals', 'symbolic_state_object_presence']
    
    embeddings = []
    labels_dict = {detector: [] for detector in detectors}
    
    for file in os.listdir(directory):
        if file.endswith(".pt") and file.startswith("episode_"):
            if exclude_files and file in exclude_files:
                continue
            file_path = os.path.join(directory, file)
            try:
                # Load the file without weights_only
                data = torch.load(
                    file_path,
                    map_location=torch.device("cpu")
                )
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            
            # Ensure the expected keys are present
            required_keys = ["visual_semantic_encoding"] + detectors
            if all(key in data for key in required_keys):
                # Extract embeddings for the specified layer
                layer_embeddings = data["visual_semantic_encoding"].get(layer_idx)
                if layer_embeddings is not None:
                    # Convert list of embeddings to NumPy array
                    layer_embeddings_np = np.array(layer_embeddings)  # Shape: (num_actions, hidden_dim)
                    embeddings.append(layer_embeddings_np)
                else:
                    print(f"Warning: Layer {layer_idx} not found in {file}. Skipping embeddings.")
                    continue  # Skip to next file
                
                # Extract and store labels for each detector
                for detector in detectors:
                    labels_np = data[detector].numpy()  # Shape: (num_actions, num_labels)
                    labels_dict[detector].append(labels_np)
            else:
                print(f"Warning: Missing keys in {file}. Skipping.")
    
    # Concatenate embeddings and labels from all files
    if embeddings:
        embeddings = np.vstack(embeddings)  # Shape: (total_actions, hidden_dim)
    else:
        embeddings = np.array([])
    
    for detector in detectors:
        if labels_dict[detector]:
            labels_dict[detector] = np.vstack(labels_dict[detector])  # Shape: (total_actions, num_labels)
        else:
            labels_dict[detector] = np.array([])
    
    return embeddings, labels_dict

def train_probe(X_train, y_train, X_test, y_test, label_names, save_path=None):
    """
    Train a probe for a single detector using multi-class multi-label classification and evaluate on test data.

    Args:
        X_train (np.ndarray): Training embeddings.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing embeddings.
        y_test (np.ndarray): Testing labels.
        label_names (list): List of label names corresponding to each column in y_train/y_test.
        save_path (str, optional): Path to save the trained probes.

    Returns:
        tuple: (f1_scores, per_label_accuracy)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss

    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Unique values in y_train: {np.unique(y_train)}")
    print(f"Unique values in y_test: {np.unique(y_test)}")

    # Initialize dictionaries to store classifiers and results
    classifiers = {}
    f1_scores = {}
    per_label_accuracy = {}

    for label_idx, label_name in enumerate(label_names):
        y_train_label = y_train[:, label_idx]
        y_test_label = y_test[:, label_idx]

        unique_classes = np.unique(y_train_label)
        print(f"\nProcessing Label: {label_name}")
        print(f"Unique classes in training data: {unique_classes}")

        if len(unique_classes) < 2:
            # Only one class present in training data
            single_class = unique_classes[0]
            print(f"Label '{label_name}' has only one class '{single_class}' in training data. Skipping classifier training.")
            # Assign the single class to all test samples
            y_pred = np.full_like(y_test_label, single_class)
            # Calculate metrics
            acc = accuracy_score(y_test_label, y_pred)
            f1 = f1_score(y_test_label, y_pred, average='macro', zero_division=0)
            f1_scores[label_name] = f1
            per_label_accuracy[label_name] = acc
            # Optionally, store a dummy classifier
            classifiers[label_name] = None
            continue

        # Initialize the classifier for multi-class classification
        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs'
        )

        # Train the classifier
        print(f"Training classifier for label '{label_name}'...")
        try:
            clf.fit(X_train, y_train_label)
            classifiers[label_name] = clf
        except ValueError as ve:
            print(f"Error training classifier for label '{label_name}': {ve}")
            classifiers[label_name] = None
            f1_scores[label_name] = np.nan
            per_label_accuracy[label_name] = np.nan
            continue

        # Predict on the test data
        y_pred = clf.predict(X_test)  # No need for mapping, predictions are already in the correct format

        # Calculate metrics
        acc = accuracy_score(y_test_label, y_pred)
        f1 = f1_score(y_test_label, y_pred, average='macro', zero_division=0)

        print(f"Accuracy for label '{label_name}': {acc * 100:.2f}%")
        print(f"F1 Score (Macro) for label '{label_name}': {f1:.4f}")

        f1_scores[label_name] = f1
        per_label_accuracy[label_name] = acc

    # Save the trained classifiers if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        joblib.dump(classifiers, save_path)
        print(f"\nAll probes saved to {save_path}\n")

    return f1_scores, per_label_accuracy

# --- Aggregate Accuracy by Category ---
def aggregate_accuracy_by_category(label_accuracies, label_names, category_mappings):
    """
    Aggregate per-label accuracies to category-level accuracies.

    Args:
        label_accuracies (dict): Dictionary of per-label accuracies.
        label_names (list): List of label names corresponding to accuracies.
        category_mappings (dict): Mapping from category names to lists of labels.

    Returns:
        dict: Mapping from category names to average accuracies.
    """
    category_accuracy = {}
    for category, labels in category_mappings.items():
        accuracies = []
        for label in labels:
            if label in label_accuracies:
                acc = label_accuracies[label]
                if not np.isnan(acc):
                    accuracies.append(acc)
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            category_accuracy[category] = avg_accuracy
        else:
            category_accuracy[category] = np.nan
            print(f"Warning: No valid accuracies found for category '{category}'.")
    return category_accuracy

# --- Function to Display Label Distribution ---
def display_label_distribution(y, label_names, threshold=None):
    """
    Compute and display the distribution of labels.
    
    Args:
        y (np.ndarray): Binary label matrix of shape (num_samples, num_labels).
        label_names (list): List of label names corresponding to each column in y.
        threshold (float, optional): If provided, filters labels with frequency below the threshold.
                                     Useful for focusing on labels that change more frequently.
    
    Returns:
        None
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("y should be a NumPy array.")
    if y.shape[1] != len(label_names):
        raise ValueError("Number of label names must match the number of label columns in y.")
    
    # Calculate frequency of each label
    frequencies = {}
    for label_idx, label_name in enumerate(label_names):
        unique_vals = np.unique(y[:, label_idx])
        frequencies[label_name] = unique_vals

    # Print the frequencies
    for label_name, unique_vals in frequencies.items():
        print(f"{label_name}: {unique_vals}")

    print("\n")  # Add space after the distribution

# --- Create Category Mappings ---
def create_category_mappings(label_names):
    """
    Create a mapping from category to labels based on the first word of each label.

    Args:
        label_names (list): List of label names.

    Returns:
        dict: Mapping from category names to lists of label names.
    """
    category_to_labels = defaultdict(list)
    for label in label_names:
        category = label.split(' ')[0]
        category_to_labels[category].append(label)
    return dict(category_to_labels)

# --- Visualization Function ---
def plot_heatmap(accuracy_matrix, categories_order, selected_layers, filename):
    """
    Plot a heatmap of category-level accuracies across selected layers and save to file.

    Args:
        accuracy_matrix (np.ndarray): Matrix of accuracies.
        categories_order (list): List of category names.
        selected_layers (list): List of layer indices.
        filename (str): Path to save the heatmap image file.
    """
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=categories_order,
        yticklabels=[f"Layer {layer}" for layer in selected_layers],
        cbar_kws={'label': 'Accuracy'},
        linewidths=.5,
        linecolor='gray'
    )
    plt.xlabel("Category")
    plt.ylabel("Layer")
    plt.title("Category-Level Accuracy Across Selected Layers")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the figure to a file
    plt.close()  # Close the figure to free memory

# --- Main Processing Function ---
def main():
    
    # Define the paths to the symbolic state key files
    object_relations_file = "/root/openvla/experiments/robot/libero/libero10_object_relations_keys.txt"
    action_subgoals_file = "/root/openvla/experiments/robot/libero/libero10_action_states_keys.txt"
    object_presence_file = "/root/openvla/experiments/robot/libero/libero10_object_presence_keys.txt"  # Third Detector

    # Read and parse object relations symbols
    with open(object_relations_file, "r") as file:
        object_relations_content = file.read().strip()
    ALL_OBJECT_RELATIONS_SYMBOLS = ast.literal_eval(object_relations_content)

    # Read and parse action subgoals symbols
    with open(action_subgoals_file, "r") as file:
        action_subgoals_content = file.read().strip()
    ALL_ACTION_SUBGOALS_SYMBOLS = ast.literal_eval(action_subgoals_content)

    # Read and parse object presence symbols (Third Detector)
    with open(object_presence_file, "r") as file:
        object_presence_content = file.read().strip()
    ALL_OBJECT_PRESENCE_SYMBOLS = ast.literal_eval(object_presence_content)

    # Create separate symbol-to-index mappings
    object_relations_symbol_to_index_libero10 = {symbol: idx for idx, symbol in enumerate(ALL_OBJECT_RELATIONS_SYMBOLS)}
    num_object_relations_labels = len(ALL_OBJECT_RELATIONS_SYMBOLS)
    # Extract the first word in each key and put them into a set
    object_relations_categories = {key.split(' ')[0] for key in object_relations_symbol_to_index_libero10.keys()}
    # Example: {'on', 'behind', 'left-of', 'right-of', 'in-front-of', 'inside', 'turned-on', 'open', 'on-table'}

    action_subgoals_symbol_to_index_libero10 = {symbol: idx for idx, symbol in enumerate(ALL_ACTION_SUBGOALS_SYMBOLS)}
    num_action_subgoals_labels = len(ALL_ACTION_SUBGOALS_SYMBOLS)
    # Extract the first word in each key and put them into a set
    action_subgoals_categories = {key.split(' ')[0] for key in action_subgoals_symbol_to_index_libero10.keys()}
    # Example: {'grasped', 'should-move-towards'}

    object_presence_symbol_to_index_libero10 = {symbol: idx for idx, symbol in enumerate(ALL_OBJECT_PRESENCE_SYMBOLS)}
    num_object_presence_labels = len(ALL_OBJECT_PRESENCE_SYMBOLS)
    # Extract the first word in each key and put them into a set
    object_presence_categories = {key.split(' ')[0] for key in object_presence_symbol_to_index_libero10.keys()}
    # Example: {'present'}

    # Combine all categories from the three sets
    all_categories = sorted(list(
        object_relations_categories.union(action_subgoals_categories, object_presence_categories)
    ))

    print(f"All Categories: {all_categories}")

    # Select the suite you are working with
    suite = 'libero-10'  # Change to 'libero-10' as needed

    if suite == 'libero-10':
        detectors = ['symbolic_state_object_relations', 'symbolic_state_action_subgoals', 'symbolic_state_object_presence']
        label_mappings = {
            'symbolic_state_object_relations': object_relations_symbol_to_index_libero10,
            'symbolic_state_action_subgoals': action_subgoals_symbol_to_index_libero10,
            'symbolic_state_object_presence': object_presence_symbol_to_index_libero10
        }
    elif suite == 'libero-spatial':
        detectors = ['symbolic_state_object_relations', 'symbolic_state_action_subgoals']
        label_mappings = {
            'symbolic_state_object_relations': object_relations_symbol_to_index_libero_spatial,
            'symbolic_state_action_subgoals': action_subgoals_symbol_to_index_libero_spatial
        }
    else:
        print("Invalid suite specified. Choose 'libero-10' or 'libero-spatial'.")
        return None, None, None

    # Create category to labels mapping for each detector
    category_mappings_per_detector = {}
    for detector in detectors:
        label_names = list(label_mappings[detector].keys())
        category_to_labels = create_category_mappings(label_names)
        category_mappings_per_detector[detector] = category_to_labels

    # Print category mappings for verification
    for detector, mapping in category_mappings_per_detector.items():
        print(f"\nCategory Mappings for {detector}:")
        for category, labels in mapping.items():
            print(f"  {category}: {labels}")

    # Define the selected layers
    selected_layers = [0, 8, 16, 24, 32]
    print(f"\nSelected Layers for Processing: {selected_layers}")

    # Define files to exclude (modify as needed)
    exclude_files = [
        'episode_5.pt', 'episode_6.pt', 'episode_19.pt', 'episode_21.pt', 'episode_22.pt', 
        'episode_24.pt', 'episode_25.pt', 'episode_26.pt', 'episode_27.pt', 'episode_30.pt', 
        'episode_35.pt', 'episode_37.pt', 'episode_38.pt', 'episode_39.pt', 'episode_46.pt', 
        'episode_47.pt', 'episode_48.pt', 'episode_50.pt', 'episode_51.pt', 'episode_52.pt', 
        'episode_53.pt', 'episode_55.pt', 'episode_56.pt', 'episode_57.pt', 'episode_58.pt', 
        'episode_59.pt', 'episode_60.pt'
    ]

    # Define data directory
    data_directory = "/root/autodl-tmp/logs_60_episodes_of_libero10"  # Update this path as needed

    # Initialize accuracy matrix for all selected layers and categories
    num_categories = len(all_categories)
    print("Number of Categories:", num_categories)
    num_selected_layers = len(selected_layers)
    accuracy_matrix = np.full((num_selected_layers, num_categories), np.nan)  # Initialize with NaN

    # Loop through each selected layer
    for idx, layer_idx in enumerate(selected_layers):
        print(f"\n=== Processing Layer {layer_idx} ===")
        
        # Load embeddings and labels for the current layer
        embeddings, labels_dict = load_probe_data(
            directory=data_directory,
            layer_idx=layer_idx,
            exclude_files=exclude_files,
            detectors=detectors
        )
        
        # Check if embeddings are loaded
        if embeddings.size == 0:
            print(f"No embeddings found for Layer {layer_idx}. Skipping.")
            continue
        
        # Iterate over each detector
        for detector in detectors:
            readable_name = detector.replace('symbolic_state_', '').replace('_', ' ').title()
            print(f"\n--- Training Probe for {readable_name} Detector in Layer {layer_idx} ---")
            
            # Extract label names
            label_names = list(label_mappings[detector].keys())
            
            # Extract labels for the current detector
            y = labels_dict[detector]
            
            if y.size == 0:
                print(f"No labels found for {detector} in Layer {layer_idx}. Skipping.")
                continue
            
            # Display label distribution
            print("Label Distribution:")
            display_label_distribution(y, label_names)
            
            # **Removed Label Filtering**
            # Use all labels without excluding any
            y_filtered = y
            label_names_filtered = label_names
            
            # Proceed without filtering
            print("Using all labels without filtering.")
            
            if y_filtered.shape[1] == 0:
                print(f"No labels available for {readable_name} in Layer {layer_idx}. Skipping probe training.")
                continue  # Skip training if no labels to train on
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, y_filtered, test_size=0.2, random_state=42
            )
            
            # Train the probe and get F1 scores and per-label accuracies
            f1_scores, per_label_accuracy = train_probe(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                label_names=label_names_filtered,
                save_path=f"savepath/probe_{detector}_layer{layer_idx}.joblib"
            )
            
            # Aggregate accuracies by category (only relevant categories)
            category_mappings = category_mappings_per_detector[detector]
            category_accuracy = aggregate_accuracy_by_category(
                label_accuracies=per_label_accuracy,
                label_names=label_names_filtered,
                category_mappings=category_mappings
            )
            
            # Update the accuracy matrix
            for category, acc in category_accuracy.items():
                if category in all_categories:
                    category_idx = all_categories.index(category)
                    if not np.isnan(accuracy_matrix[idx, category_idx]):
                        # Instead of taking the maximum, average the existing and new accuracy
                        existing_acc = accuracy_matrix[idx, category_idx]
                        accuracy_matrix[idx, category_idx] = (existing_acc + acc) / 2
                    else:
                        accuracy_matrix[idx, category_idx] = acc
                else:
                    print(f"Category '{category}' not found in all_categories.")
    return accuracy_matrix, selected_layers, all_categories

# --- Visualization Function ---
def plot_heatmap(accuracy_matrix, categories_order, selected_layers, filename):
    """
    Plot a heatmap of category-level accuracies across selected layers and save to file.

    Args:
        accuracy_matrix (np.ndarray): Matrix of accuracies.
        categories_order (list): List of category names.
        selected_layers (list): List of layer indices.
        filename (str): Path to save the heatmap image file.
    """
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=categories_order,
        yticklabels=[f"Layer {layer}" for layer in selected_layers],
        cbar_kws={'label': 'Accuracy'},
        linewidths=.5,
        linecolor='gray'
    )
    plt.xlabel("Category")
    plt.ylabel("Layer")
    plt.title("Category-Level Accuracy Across Selected Layers")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the figure to a file
    plt.close()  # Close the figure to free memory

# --- Execute the main function ---
if __name__ == "__main__":
    # Run the main analysis
    accuracy_matrix, selected_layers, categories_order = main()
    print("Categories Order:", categories_order)
    
    # Plot results if available
    if accuracy_matrix is not None and selected_layers is not None and categories_order is not None:
        plot_heatmap(accuracy_matrix, categories_order, selected_layers, "heatmap.png")
        print("Heatmap saved as 'heatmap.png'.")
    else:
        print("Accuracy matrix or other necessary data not available for plotting.")
