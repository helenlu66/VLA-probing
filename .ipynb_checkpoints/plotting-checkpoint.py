import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example data:
# accuracies[i, j] = accuracy of the i-th classifier on the j-th category
accuracies = np.array([
    [0.85, 0.90, 0.78, 0.83],
    [0.88, 0.92, 0.80, 0.81],
    [0.82, 0.91, 0.77, 0.84]
])

layer_indices = ['Layer 0', 'Layer 4', 'Layer 8', 'Layer 12', 'Layer 16', 'Layer 20', 'Layer 24', 'Layer 28', 'Layer 32']
category_names = ['o', 'Category 2', 'Category 3', 'Category 4']

# Create the heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(accuracies, 
                 annot=True,        # Show values in each cell
                 fmt=".2f",         # Format numbers as floats with 2 decimals
                 cmap="Blues",       # Choose a color palette
                 xticklabels=category_names, 
                 yticklabels=classifier_names)

ax.set_xlabel("Category")
ax.set_ylabel("Classifier")
ax.set_title("Classification Accuracy Heatmap")

plt.tight_layout()
plt.show()
