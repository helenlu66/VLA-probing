import os
import joblib
import torch
import numpy as np

# --- Step 1: Load the Probe ---
probe_path = "probe_symbolic_state_action_subgoals_layer32.joblib"  # Replace with your actual path
probe = load_probe(probe_path)

if probe is None:
    raise ValueError("Failed to load the probe. Aborting inference.")

# --- Step 2: Load New Embeddings ---
new_data_directory = "/path/to/new/episode_files"  # Replace with your data directory
layer_idx = 32  # Replace with the layer used during training

new_embeddings = load_new_embeddings(directory=new_data_directory, layer_idx=layer_idx)

if new_embeddings.size == 0:
    raise ValueError("No embeddings found for inference.")

print(f"Loaded new embeddings with shape: {new_embeddings.shape}")

# --- Step 3: Perform Inference ---
predictions = perform_inference(pipeline=probe, embeddings=new_embeddings)

if predictions is None:
    raise ValueError("Inference failed.")

print(f"Predictions shape: {predictions.shape}")
print("Sample Predictions:")
print(predictions[:5])  # Print first 5 predictions

# --- Step 4: Map Predictions to Labels ---
# Load label names (ensure they match the order used during training)
# This assumes you have access to the same label mappings used during training
# For example, if you trained on 'symbolic_state_object_relations':
object_relations_file = "/root/openvla/experiments/robot/libero/spatial_object_relations_keys.txt"

with open(object_relations_file, "r") as file:
    object_relations_content = file.read().strip()
ALL_OBJECT_RELATIONS_SYMBOLS = ast.literal_eval(object_relations_content)

label_names = list(ALL_OBJECT_RELATIONS_SYMBOLS)  # Ensure this matches the order during training

mapped_predictions = map_predictions_to_labels(predictions, label_names)

# --- Step 5: Inspect Predictions ---
for i, labels in enumerate(mapped_predictions[:5]):
    print(f"Sample {i+1}: {labels}")
