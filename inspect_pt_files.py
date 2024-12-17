import torch

# Example: Inspect a single .pt file
sample_file = "/root/autodl-tmp/logs_60_episodes_of_libero10/episode_1.pt"
data = torch.load(sample_file, map_location=torch.device("cpu"))
print(data.keys())
print(data['symbolic_state_object_relations'].shape)  # Should match the number of object relations labels
print(data['symbolic_state_action_subgoals'].shape)  # Should match the number of action subgoals labels
print(data['symbolic_state_object_presence'].shape)  # Should match the number of object presence labels
