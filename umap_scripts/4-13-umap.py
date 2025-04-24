# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import umap

dfs_and_names = [
    (
        "ground-truth",
        pd.read_pickle("ground_truth_with_embeddings.pkl"),
    ),
    (
        "crystal-text-llm",
        pd.read_pickle("./data/crystal_text_llm_mp20_test_with_embeddings.pkl"),
    ),
    (
        "cdvae",
        pd.read_pickle("./data/cdvae_mp20_test_with_embeddings.pkl"),
    ),
    (
        "flowmm",
        pd.read_pickle("./data/flowmm_mp20_test_with_embeddings.pkl"),
    ),
]


def add_umap_to_dataframe(
    df: pd.DataFrame,
    embeddings_col="embeddings",
    n_components=2,
):
    """
    Apply UMAP dimensionality reduction on embeddings and add the results as new columns to the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the embeddings column
    embeddings_col : str, optional
        Name of the column containing embeddings, default is 'embeddings'
    n_components : int, optional
        Number of UMAP dimensions, default is 2
    n_neighbors : int, optional
        Number of neighbors to consider, controls local vs global structure (lower values = more local clustering)
    min_dist : float, optional
        Minimum distance parameter for UMAP, lower values increase clustering
    metric : str, optional
        Distance metric to use, default is 'euclidean'

    Returns
    -------
    pandas.DataFrame
        DataFrame with added UMAP columns
    """
    # Extract embeddings into a single array
    embeddings_array = np.vstack(df[embeddings_col].tolist())

    # Apply UMAP
    print(
        f"Running UMAP on {embeddings_array.shape[0]} samples with dimension {embeddings_array.shape[1]}..."
    )
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    umap_results = reducer.fit_transform(embeddings_array)
    print("UMAP completed.", umap_results.shape)

    # Add results to dataframe
    df = df.copy()
    for i in range(n_components):
        df[f"umap_{i + 1}"] = umap_results[:, i]
        df[f"umap_{i + 1}"] = df[f"umap_{i + 1}"].astype(float)

    # Drop the original embeddings column
    df = df.drop(columns=[embeddings_col])

    print(
        f"UMAP completed. Added columns: {', '.join([f'umap_{i + 1}' for i in range(n_components)])}"
    )
    return df


dfs_list = [df for _, df in dfs_and_names]
subsample_size: int | None = 5000
# in case the DFs are different size, we can subsample all to the smallest size
min_size = min([len(df) for df in dfs_list])
if subsample_size is not None:
    min_size = min(subsample_size, min_size)

if not all([len(df) == min_size for df in dfs_list]):
    print("Subsampling all dataframes to the smallest size:", min_size)
    dfs_list = [
        df.sample(min_size, random_state=42).reset_index(drop=True) for df in dfs_list
    ]

df = pd.concat(dfs_list, ignore_index=True).reset_index(drop=True)


# Apply UMAP with parameters for better clustering
df = add_umap_to_dataframe(df)

# rename umap columns to 2d_x, 2d_y
df = df.rename(columns={"umap_1": "2d_x", "umap_2": "2d_y"})
df.to_pickle(saved_path := "../tsne-final.pkl")
print(f"Saved to {saved_path}")
df


import matplotlib.pyplot as plt
import seaborn as sns


def draw_plot(df: pd.DataFrame, title: str = "UMAP Plot"):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="2d_x",
        y="2d_y",
        hue="model",
        data=df,
        palette="viridis",
        alpha=0.7,
        edgecolor=None,
    )
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()


draw_plot(df)

# %%
