import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import ast
from tqdm import tqdm

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

# --- Data Loading Function with Episode-Level Splitting ---
def load_probe_data_episode_split(directory, exclude_files=None, detectors=None, layer_idx=None):
    """
    Load embeddings and labels for a specific layer from .pt files, then split episodes into train and test sets.
    
    Args:
        directory (str): Path to the directory containing .pt files.
        exclude_files (list, optional): List of filenames to exclude.
        detectors (list, optional): List of detector keys to extract labels for.
        layer_idx (int, optional): Specific layer index to load embeddings for.
    
    Returns:
        tuple: (train_embeddings, train_labels_dict, test_embeddings, test_labels_dict)
    """
    if detectors is None:
        detectors = ['symbolic_state_object_relations', 'symbolic_state_action_subgoals']
    
    episode_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(".pt") and f.startswith("episode_") and (exclude_files is None or f not in exclude_files)
    ]
    
    if not episode_files:
        raise ValueError("No episode files found in the specified directory.")
    
    # Split episodes into train and test
    train_files, test_files = train_test_split(episode_files, test_size=0.2, random_state=42)
    
    def load_files(file_list):
        embeddings = []
        labels_dict = {detector: [] for detector in detectors}
        for file in file_list:
            try:
                data = torch.load(file, map_location=torch.device("cpu"))
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            # Ensure the expected keys are present
            required_keys = ["visual_semantic_encoding"] + detectors
            if all(key in data for key in required_keys):
                for detector in detectors:
                    if detector not in data:
                        print(f"Warning: Detector '{detector}' not found in {file}. Skipping.")
                        continue

                # Iterate through each layer if multiple layers are stored
                layer_embeddings_dict = data["visual_semantic_encoding"]
                
                if layer_idx is not None:
                    # Load only the specified layer
                    if layer_idx not in layer_embeddings_dict:
                        print(f"Warning: Layer {layer_idx} not found in {file}. Skipping this file.")
                        continue
                    current_layer_embeddings = layer_embeddings_dict[layer_idx]
                    if current_layer_embeddings is None:
                        print(f"Warning: Layer {layer_idx} data is None in {file}. Skipping.")
                        continue

                    # Handle different types of layer_embeddings
                    if isinstance(current_layer_embeddings, torch.Tensor):
                        embeddings_np = current_layer_embeddings.numpy()
                    elif isinstance(current_layer_embeddings, list):
                        embeddings_np = np.array(current_layer_embeddings)
                    else:
                        print(f"Unexpected type for layer_embeddings in {file}, layer {layer_idx}: {type(current_layer_embeddings)}. Skipping.")
                        continue

                    embeddings.append(embeddings_np)  # Shape: (num_actions, hidden_dim)
                else:
                    # Load all layers (if needed)
                    for layer_i, layer_embeddings in layer_embeddings_dict.items():
                        if layer_embeddings is None:
                            print(f"Warning: Layer {layer_i} not found in {file}. Skipping.")
                            continue

                        # Handle different types of layer_embeddings
                        if isinstance(layer_embeddings, torch.Tensor):
                            embeddings_np = layer_embeddings.numpy()
                        elif isinstance(layer_embeddings, list):
                            embeddings_np = np.array(layer_embeddings)
                        else:
                            print(f"Unexpected type for layer_embeddings in {file}, layer {layer_i}: {type(layer_embeddings)}. Skipping.")
                            continue

                        embeddings.append(embeddings_np)  # Shape: (num_actions, hidden_dim)
                
                for detector in detectors:
                    label_data = data.get(detector)
                    if label_data is None:
                        print(f"Warning: Detector '{detector}' data missing in {file}. Skipping labels for this detector.")
                        continue

                    # Handle different types of label_data
                    if isinstance(label_data, torch.Tensor):
                        labels_np = label_data.numpy()
                    elif isinstance(label_data, list):
                        labels_np = np.array(label_data)
                    else:
                        print(f"Unexpected type for labels in {file}, detector {detector}: {type(label_data)}. Skipping labels for this detector.")
                        continue

                    labels_dict[detector].append(labels_np)
            else:
                print(f"Warning: Missing keys in {file}. Required keys: {required_keys}. Skipping.")

        # Concatenate embeddings and labels from all episodes
        if embeddings:
            embeddings = np.vstack(embeddings)  # Shape: (total_actions * layers, hidden_dim)
        else:
            embeddings = np.array([])

        for detector in detectors:
            if labels_dict[detector]:
                labels_dict[detector] = np.vstack(labels_dict[detector])  # Shape: (total_actions, num_labels)
            else:
                labels_dict[detector] = np.array([])

        return embeddings, labels_dict

    # Load training data
    train_embeddings, train_labels_dict = load_files(train_files)
    # Load testing data
    test_embeddings, test_labels_dict = load_files(test_files)

    return train_embeddings, train_labels_dict, test_embeddings, test_labels_dict

# --- Probe Training Function ---
def train_probe(X_train, y_train, X_test, y_test, label_names, save_path=None):
    """
    Train a probe for a single detector using multi-label classification and evaluate on test data.

    Args:
        X_train (np.ndarray): Training embeddings.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing embeddings.
        y_test (np.ndarray): Testing labels.
        label_names (list): List of label names corresponding to each column in y_train/y_test.
        save_path (str, optional): Path to save the trained probe.

    Returns:
        tuple: (f1_scores, per_label_accuracy)
    """
    # Initialize the Logistic Regression classifier wrapped in OneVsRestClassifier
    base_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf = OneVsRestClassifier(base_clf)

    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    # Train the classifier
    print("Training the classifier...")
    pipeline.fit(X_train, y_train)

    # Predict on the test data
    y_pred = pipeline.predict(X_test)

    # Evaluation Metrics
    print("\n=== Evaluation Metrics ===")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Subset Accuracy: {accuracy * 100:.2f}%")  # Exact match ratio

    hamming = hamming_loss(y_test, y_pred)
    print(f"Hamming Loss: {hamming:.4f}")

    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}\n")

    # Detailed Classification Report with zero_division set to 0 to suppress warnings
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    # Compute F1 scores for each label
    f1_scores = f1_score(y_test, y_pred, average=None)

    # Compute per-label accuracy manually
    per_label_accuracy = np.mean(y_pred == y_test, axis=0)  # Accuracy per label

    # Save the trained classifier if a path is provided
    if save_path:
        joblib.dump(pipeline, save_path)
        print(f"Probe saved to {save_path}\n")

    return f1_scores, per_label_accuracy

# --- Helper Function to Identify Low Variance Labels ---
def identify_low_variance_labels(y, label_names, threshold=0.001):
    """
    Identify labels that are constant or change below a specified frequency threshold.
    
    Args:
        y (np.ndarray): Label array, shape (num_samples, num_labels).
        label_names (list): List of label names corresponding to each column in y.
        threshold (float): Minimum fraction of samples that must have label=1 to be retained.
                           Labels with frequency below this threshold are considered low variance.
    
    Returns:
        list: Names of labels that are constant or have low variance.
    """
    low_variance_labels = []
    for idx, label in enumerate(label_names):
        label_sum = np.sum(y[:, idx])
        frequency = label_sum / y.shape[0]
        if frequency == 0.0 or frequency == 1.0:
            # Constant label
            low_variance_labels.append(label)
        elif frequency < threshold or frequency > (1 - threshold):
            # Low frequency label (very few 1s or very few 0s)
            low_variance_labels.append(label)
    return low_variance_labels

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
    frequencies = np.mean(y, axis=0)  # Proportion of 1s for each label
    
    # Create a list of tuples (label_name, frequency)
    label_freq = list(zip(label_names, frequencies))
    
    # Sort labels by frequency in descending order
    label_freq_sorted = sorted(label_freq, key=lambda x: x[1], reverse=True)
    
    # If a threshold is provided, filter labels below the threshold
    if threshold is not None:
        label_freq_sorted = [lf for lf in label_freq_sorted if lf[1] >= threshold]
        print(f"Labels with frequency >= {threshold*100:.2f}%:")
    else:
        print("All Label Distributions:")
    
    # Print the sorted label frequencies
    for label, freq in label_freq_sorted:
        percentage = freq * 100
        status = ""
        if freq == 0.0 or freq == 1.0:
            status = "(Constant)"
        elif freq < 0.01 or freq > 0.99:
            status = "(Low Variance)"
        print(f"{label}: {percentage:.2f}% {status}")
    
    print("\n")  # Add space after the distribution

# --- Function to Aggregate Accuracy by Category ---
def aggregate_accuracy_by_category(label_accuracies, label_names, category_mappings):
    """
    Aggregate per-label accuracies to category-level accuracies.

    Args:
        label_accuracies (np.ndarray): Array of per-label accuracies.
        label_names (list): List of label names corresponding to accuracies.
        category_mappings (dict): Mapping from category names to lists of labels.

    Returns:
        dict: Mapping from category names to average accuracies.
    """
    category_accuracy = {}
    for category, labels in category_mappings.items():
        # Find indices of labels belonging to this category
        label_indices = [i for i, label in enumerate(label_names) if label in labels]
        if not label_indices:
            category_accuracy[category] = np.nan  # or 0, or skip
            print(f"Warning: No labels found for category '{category}'.")
            continue
        # Compute average accuracy for the category
        avg_accuracy = np.mean(label_accuracies[label_indices])
        category_accuracy[category] = avg_accuracy
    return category_accuracy

# --- Function to Create Category Mappings ---
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
def plot_revised_heatmap(accuracy_matrix, categories_order, selected_layers, exclude_categories, move_to_end, filename="revised_heatmap.png"):
    """
    Plot a heatmap of category-level accuracies across selected layers with specified exclusions and reordering.

    Args:
        accuracy_matrix (np.ndarray): Matrix of accuracies with shape (num_layers, num_categories).
        categories_order (list): List of category names corresponding to the columns of accuracy_matrix.
        selected_layers (list): List of layer indices corresponding to the rows of accuracy_matrix.
        exclude_categories (list): List of category names to exclude from the heatmap.
        move_to_end (list): List of category names to move to the end of the heatmap.
        filename (str): Path to save the heatmap image file.

    Returns:
        None
    """
    # Step 1: Exclude specified categories
    indices_to_keep = [i for i, cat in enumerate(categories_order) if cat not in exclude_categories]
    filtered_categories = [cat for cat in categories_order if cat not in exclude_categories]
    filtered_accuracy_matrix = accuracy_matrix[:, indices_to_keep]

    # Step 2: Reorder categories to move specified ones to the end
    # Identify the categories to move and their indices
    move_categories_present = [cat for cat in move_to_end if cat in filtered_categories]
    categories_remaining = [cat for cat in filtered_categories if cat not in move_categories_present]
    new_categories_order = categories_remaining + move_categories_present

    # Reorder the columns in the accuracy matrix accordingly
    new_order_indices = [filtered_categories.index(cat) for cat in new_categories_order]
    reordered_accuracy_matrix = filtered_accuracy_matrix[:, new_order_indices]

    # Step 3: Plot the heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        reordered_accuracy_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=new_categories_order,
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
    print(f"Heatmap saved to {filename}")

# --- Main Function ---
def main():
    # Define the paths to the symbolic state key files
    object_relations_file = "/root/openvla/experiments/robot/libero/spatial_object_relations_keys.txt"
    action_subgoals_file = "/root/openvla/experiments/robot/libero/spatial_action_states_keys.txt"
    
    # Read and parse object relations symbols
    with open(object_relations_file, "r") as file:
        object_relations_content = file.read().strip()
    ALL_OBJECT_RELATIONS_SYMBOLS = ast.literal_eval(object_relations_content)
    
    # Read and parse action subgoals symbols
    with open(action_subgoals_file, "r") as file:
        action_subgoals_content = file.read().strip()
    ALL_ACTION_SUBGOALS_SYMBOLS = ast.literal_eval(action_subgoals_content)
    
    # Create separate symbol-to-index mappings
    object_relations_symbol_to_index_libero_spatial = {symbol: idx for idx, symbol in enumerate(ALL_OBJECT_RELATIONS_SYMBOLS)}
    num_object_relations_labels = len(ALL_OBJECT_RELATIONS_SYMBOLS)
    # Extract the first word in each key and put them into a set
    object_relations_categories = {key.split(' ')[0] for key in object_relations_symbol_to_index_libero_spatial.keys()}
    # Example: {'on', 'behind', 'left-of', 'right-of', 'in-front-of', 'inside', 'turned-on', 'open', 'on-table'}
    
    action_subgoals_symbol_to_index_libero_spatial = {symbol: idx for idx, symbol in enumerate(ALL_ACTION_SUBGOALS_SYMBOLS)}
    num_action_subgoals_labels = len(ALL_ACTION_SUBGOALS_SYMBOLS)
    # Extract the first word in each key and put them into a set
    action_subgoals_categories = {key.split(' ')[0] for key in action_subgoals_symbol_to_index_libero_spatial.keys()}
    # Example: {'grasped', 'should-move-towards'}
    
    # Combine all categories
    all_categories = sorted(list(object_relations_categories.union(action_subgoals_categories)))
    print(f"All Categories: {all_categories}")
    
    # Select the suite you are working with
    suite = 'libero_spatial'  # Change to 'libero_10' as needed
    
    if suite == 'libero_10':
        # Define label mappings for 'libero_10' (Assuming these dictionaries are defined elsewhere)
        # Example:
        # object_relations_symbol_to_index_libero10 = {...}
        # action_subgoals_symbol_to_index_libero10 = {...}
        # object_presence_symbol_to_index_libero10 = {...}
        # Replace with actual mappings
        detectors = ['symbolic_state_object_relations', 'symbolic_state_action_subgoals', 'symbolic_state_object_presence']
        # Example dictionaries (replace with actual mappings)
        object_relations_symbol_to_index_libero10 = {
            # 'relation_symbol': index,
            # ...
        }
        action_subgoals_symbol_to_index_libero10 = {
            # 'action_symbol': index,
            # ...
        }
        object_presence_symbol_to_index_libero10 = {
            # 'presence_symbol': index,
            # ...
        }
        label_mappings = {
            'symbolic_state_object_relations': object_relations_symbol_to_index_libero10,
            'symbolic_state_action_subgoals': action_subgoals_symbol_to_index_libero10,
            'symbolic_state_object_presence': object_presence_symbol_to_index_libero10
        }
    elif suite == 'libero_spatial':
        detectors = ['symbolic_state_object_relations', 'symbolic_state_action_subgoals']
        label_mappings = {
            'symbolic_state_object_relations': object_relations_symbol_to_index_libero_spatial,
            'symbolic_state_action_subgoals': action_subgoals_symbol_to_index_libero_spatial
        }
    else:
        print("Invalid suite specified. Choose 'libero_10' or 'libero_spatial'.")
        return
    
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
    
    # Define categories order for consistency in the heatmap
    categories_order = all_categories  # ['behind', 'grasped', 'in-front-of', ...]
    
    # Define the selected layers
    selected_layers = [0, 8, 16, 24, 32]
    print(f"\nSelected Layers for Processing: {selected_layers}")
    
    # Define files to exclude (modify as needed)
    exclude_files = [
        'episode_5.pt', 'episode_6.pt', 'episode_19.pt', 'episode_21.pt', 'episode_22.pt', 'episode_24.pt',
        'episode_25.pt', 'episode_26.pt', 'episode_27.pt', 'episode_30.pt', 'episode_35.pt',
        'episode_37.pt', 'episode_38.pt', 'episode_39.pt', 'episode_46.pt', 'episode_47.pt',
        'episode_48.pt', 'episode_50.pt', 'episode_51.pt', 'episode_52.pt', 'episode_53.pt',
        'episode_55.pt', 'episode_56.pt', 'episode_57.pt', 'episode_58.pt', 'episode_59.pt',
        'episode_60.pt'
    ]
    
    # Define data directory
    data_directory = "/root/autodl-tmp/logs_60_episodes_of_spatial_new"  # Update this path as needed
    
    # Initialize accuracy matrix for all selected layers and categories
    num_categories = len(categories_order)
    num_selected_layers = len(selected_layers)
    accuracy_matrix = np.full((num_selected_layers, num_categories), np.nan)  # Initialize with NaN
    
    # Loop through each selected layer
    for idx, layer_idx in enumerate(selected_layers):
        print(f"\n=== Processing Layer {layer_idx} ===")
        
        # Split episodes into train and test for the specific layer
        try:
            train_embeddings, train_labels_dict, test_embeddings, test_labels_dict = load_probe_data_episode_split(
                directory=data_directory,
                exclude_files=exclude_files,
                detectors=detectors,
                layer_idx=layer_idx  # Specify the current layer
            )
        except ValueError as ve:
            print(f"Error during data loading: {ve}")
            continue
        
        # Check if embeddings are loaded
        if train_embeddings.size == 0 or test_embeddings.size == 0:
            print(f"No embeddings found for Layer {layer_idx}. Skipping.")
            continue
        
        # Verify that the number of samples in embeddings and labels match
        print(f"Training Embeddings Shape: {train_embeddings.shape}")
        for detector in detectors:
            y_train = train_labels_dict[detector]
            y_test = test_labels_dict[detector]
            print(f"Training Labels for {detector} Shape: {y_train.shape}")
            print(f"Testing Labels for {detector} Shape: {y_test.shape}")
        
        # Iterate over each detector
        for detector in detectors:
            readable_name = detector.replace('symbolic_state_', '').replace('_', ' ').title()
            print(f"\n--- Training Probe for {readable_name} Detector in Layer {layer_idx} ---")
            
            # Extract label names
            label_names = list(label_mappings[detector].keys())
            
            # Extract labels for the current detector
            y_train = train_labels_dict[detector]
            y_test = test_labels_dict[detector]
            
            if y_train.size == 0 or y_test.size == 0:
                print(f"No labels found for {detector} in Layer {layer_idx}. Skipping.")
                continue
            
            # Display label distribution before filtering
            print("Label Distribution Before Filtering:")
            display_label_distribution(y_train, label_names)
            
            # Identify and exclude constant or low variance labels
            low_variance_labels = identify_low_variance_labels(y_train, label_names, threshold=0.001)  # 0.1% threshold
            if low_variance_labels:
                print(f"Excluding constant or low variance labels: {low_variance_labels}")
                print(f"Excluding {len(low_variance_labels)} constant or low variance labels")
                # Get indices of non-excluded labels
                non_excluded_indices = [i for i, label in enumerate(label_names) if label not in low_variance_labels]
                # Update y and label_names
                y_train_filtered = y_train[:, non_excluded_indices]
                y_test_filtered = y_test[:, non_excluded_indices]
                label_names_filtered = [label for label in label_names if label not in low_variance_labels]
            else:
                print("No constant or low variance labels detected.")
                y_train_filtered = y_train
                y_test_filtered = y_test
                label_names_filtered = label_names
            
            # Display label distribution after filtering
            print("Label Distribution After Filtering:")
            display_label_distribution(y_train_filtered, label_names_filtered)
            
            if y_train_filtered.shape[1] == 0:
                print(f"All labels for {readable_name} in Layer {layer_idx} are constant or have low variance. Skipping probe training.")
                continue  # Skip training if no labels to train on
            
            # Verify that X_train and y_train have the same number of samples
            if train_embeddings.shape[0] != y_train_filtered.shape[0]:
                print(f"Mismatch in samples for {readable_name} Detector in Layer {layer_idx}:")
                print(f"X_train samples: {train_embeddings.shape[0]}, y_train samples: {y_train_filtered.shape[0]}")
                print("Skipping this detector for the current layer.")
                continue
            
            # Train the probe and get F1 scores and per-label accuracies
            f1_scores, per_label_accuracy = train_probe(
                X_train=train_embeddings,
                y_train=y_train_filtered,
                X_test=test_embeddings,
                y_test=y_test_filtered,
                label_names=label_names_filtered,
                save_path=f"probe_{detector}_layer{layer_idx}.joblib"
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
                if category in categories_order:
                    category_idx = categories_order.index(category)
                    if not np.isnan(accuracy_matrix[idx, category_idx]):
                        # Instead of taking the maximum, average the existing and new accuracy
                        existing_acc = accuracy_matrix[idx, category_idx]
                        accuracy_matrix[idx, category_idx] = (existing_acc + acc) / 2
                    else:
                        accuracy_matrix[idx, category_idx] = acc
                else:
                    print(f"Category '{category}' not found in categories_order.")
    
    return accuracy_matrix, selected_layers, categories_order

# --- Execute the main function ---
if __name__ == "__main__":
    # Run the main analysis
    accuracy_matrix, selected_layers, categories_order = main()
    
    # Define categories to exclude and reorder
    exclude_categories = ['turned-on', 'on-table']
    move_to_end = ['grasped', 'should-move-towards']
    
    # Plot the revised heatmap if data is available
    if accuracy_matrix is not None and selected_layers is not None and categories_order is not None:
        plot_revised_heatmap(
            accuracy_matrix=accuracy_matrix,
            categories_order=categories_order,
            selected_layers=selected_layers,
            exclude_categories=exclude_categories,
            move_to_end=move_to_end,
            filename="heatmap-spatial-revised.png"  # Specify your desired filename
        )
    else:
        print("Accuracy matrix or other necessary data not available for plotting.")
