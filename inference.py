import os
import joblib
import torch
import numpy as np
import ast
import time

# --- Helper Functions ---

def load_probe(probe_path):
    try:
        pipeline = joblib.load(probe_path)
        print(f"Successfully loaded probe from {probe_path}")
        return pipeline
    except Exception as e:
        print(f"Error loading probe from {probe_path}: {e}")
        return None

def load_label_names(detector, label_file_path):
    with open(label_file_path, "r") as file:
        label_content = file.read().strip()
    all_symbols = ast.literal_eval(label_content)
    return list(all_symbols)

def perform_single_inference(pipeline, embedding, label_names):
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)  # Reshape to (1, hidden_dim)

    try:
        prediction = pipeline.predict(embedding)  # Shape: (1, num_labels)
        prediction = prediction[0]  # Get the first (and only) prediction
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    # Map predictions to label names
    predicted_labels = [label for label, pred in zip(label_names, prediction) if pred == 1]

    return predicted_labels

def real_time_inference_loop(probe, label_names, embedding_generator):
    for embedding in embedding_generator:
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        elif isinstance(embedding, list):
            embedding = np.array(embedding)
        elif not isinstance(embedding, np.ndarray):
            print(f"Unsupported embedding type: {type(embedding)}. Skipping.")
            continue

        predicted_labels = perform_single_inference(probe, embedding, label_names)
        yield (embedding, predicted_labels)

# --- Main Inference Script ---

def main():
    # --- Configuration ---
    probe_path = "probe_symbolic_state_action_subgoals_layer32.joblib"  # Update path
    detector = 'symbolic_state_action_subgoals'
    label_file_path = "/root/openvla/experiments/robot/libero/spatial_action_subgoals_keys.txt"  # Update path

    # --- Load Probe ---
    probe = load_probe(probe_path)
    if probe is None:
        raise ValueError("Failed to load the probe. Aborting inference.")

    # --- Load Label Names ---
    label_names = load_label_names(detector, label_file_path)
    print(f"Loaded {len(label_names)} labels for detector '{detector}'.")

    # --- Define Embedding Generator ---
    # Replace this with your actual embedding source
    def actual_embedding_generator():
        """
        Replace this with your actual embedding generation mechanism.
        This is a placeholder that simulates real-time embedding generation.
        """
        hidden_dim = probe.named_steps['clf'].estimators_[0].coef_.shape[1]
        while True:
            # Replace the following line with actual embedding retrieval logic
            new_embedding = np.random.randn(hidden_dim)  # Example embedding
            yield new_embedding
            time.sleep(1)  # Simulate delay between actions

    # Initialize the generator
    embedding_gen = actual_embedding_generator()

    # --- Start Real-Time Inference ---
    print("Starting real-time inference. Press Ctrl+C to stop.")
    try:
        for idx, (embedding, labels) in enumerate(real_time_inference_loop(probe, label_names, embedding_gen)):
            print(f"Action {idx+1}: Predicted Labels: {labels}")
            # Optionally, implement logic based on predictions
    except KeyboardInterrupt:
        print("\nReal-time inference stopped.")

if __name__ == "__main__":
    main()
