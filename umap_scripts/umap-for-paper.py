# %%
from __future__ import annotations

import pandas as pd

df = pd.read_pickle("../tsne-final.pkl")
df
# %%

model_name_mapping = {
    "flowmm": "FlowMM",
    "cdvae": "CDVAE",
    "crystal-text-llm": "CrystalTextLLM",
    "ground_truth_with_embeddings": "Ground Truth",
}
df["Model"] = df["model"].map(model_name_mapping)

plot_labels = {
    "model": "Model",
}


# %%
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

colors = {}
for model in model_name_mapping.values():
    # Generate a color using the "Set1" colormap
    color = cm.get_cmap("Set1")(list(model_name_mapping.values()).index(model))
    # Convert to RGBA format
    colors[model] = color

# Set the style of seaborn with larger font scale
sns.set_theme(style="white", context="paper", font_scale=1.8)

# Increase matplotlib font sizes globally
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 16

# Create a scatter plot with "2d_x", "2d_y" as coordinates and "model" as the hue
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    data=df,
    x="2d_x",
    y="2d_y",
    hue="Model",
    palette=colors,
    alpha=0.5,  # Transparency
)
# remove x and y ticks, x and y labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

# Create custom legend with full opacity
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=colors[model],
        markersize=12,
        alpha=1.0,
        label=model,
    )
    for model in model_name_mapping.values()
]

# Add the custom legend
ax.legend(handles=legend_elements, fontsize=16, title_fontsize=18)

# plt.tight_layout()
plt.savefig("umap_projection.pdf", bbox_inches="tight")
