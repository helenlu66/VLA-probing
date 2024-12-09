import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example data:
# accuracies[i, j] = accuracy of the i-th classifier on the j-th category
dummy_accuracies = np.array([
    [0.85, 0.90],
    [0.88, 0.92],
    [0.82, 0.91],
    [0.85, 0.90],
    [0.88, 0.92],
    [0.82, 0.91],
    [0.85, 0.90],
    [0.88, 0.92],
    [0.82, 0.91],
])

layer_indices = ['Layer 0', 'Layer 4', 'Layer 8', 'Layer 12', 'Layer 16', 'Layer 20', 'Layer 24', 'Layer 28', 'Layer 32']
obj_relation_symbols = ['behind', 'in-front-of', 'inside', 'left-of', 'on', 'open', 'right-of', 'turned-on']
action_state_symbols = ['grasped', 'should-move-towards']

def heatmap(accuracies, obj_relations=True):
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    if obj_relations:
        ax = sns.heatmap(accuracies, 
                        annot=True,        # Show values in each cell
                        fmt=".2f",         # Format numbers as floats with 2 decimals
                        cmap="Blues",       # Choose a color palette
                        xticklabels=obj_relation_symbols, 
                        yticklabels=layer_indices)

        ax.set_xlabel("Object Relations")
        ax.set_ylabel("Hidden Layer")
        ax.set_title("Probe Accuracy Heatmap")

        plt.tight_layout()
        plt.savefig('obj_relations_heatmap.pdf')
    else:
        ax = sns.heatmap(accuracies, 
                        annot=True,        # Show values in each cell
                        fmt=".2f",         # Format numbers as floats with 2 decimals
                        cmap="Blues",       # Choose a color palette
                        xticklabels=action_state_symbols, 
                        yticklabels=layer_indices)

        ax.set_xlabel("Action States")
        ax.set_ylabel("Hidden Layer")
        ax.set_title("Probe Accuracy Heatmap")

        plt.tight_layout()
        plt.savefig('action_states_heatmap.png')
    plt.show()

if __name__ == "__main__":
    heatmap(accuracies=dummy_accuracies, obj_relations=False)