# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

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


def add_tsne_to_dataframe(
    df: pd.DataFrame,
    embeddings_col="embeddings",
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
):
    """
    Apply t-SNE dimensionality reduction on embeddings and add the results as new columns to the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the embeddings column
    embeddings_col : str, optional
        Name of the column containing embeddings, default is 'embeddings'
    n_components : int, optional
        Number of t-SNE dimensions, default is 2
    perplexity : float, optional
        Perplexity parameter for t-SNE, default is 30
    learning_rate : float, optional
        Learning rate for t-SNE, default is 200
    n_iter : int, optional
        Number of iterations for t-SNE, default is 1000

    Returns
    -------
    pandas.DataFrame
        DataFrame with added t-SNE columns
    """
    # Extract embeddings into a single array
    embeddings_array = np.vstack(df[embeddings_col].tolist())

    # Apply t-SNE
    print(
        f"Running t-SNE on {embeddings_array.shape[0]} samples with dimension {embeddings_array.shape[1]}..."
    )
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric="cosine",
        random_state=42,
    )
    tsne_results = tsne.fit_transform(embeddings_array)
    print("t-SNE completed.", tsne_results.shape)

    # Add results to dataframe
    df = df.copy()
    for i in range(n_components):
        df[f"tsne_{i + 1}"] = tsne_results[:, i]
        df[f"tsne_{i + 1}"] = df[f"tsne_{i + 1}"].astype(float)

    # Drop the original embeddings column
    df = df.drop(columns=[embeddings_col])

    print(
        f"t-SNE completed. Added columns: {', '.join([f'tsne_{i + 1}' for i in range(n_components)])}"
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

# Apply t-SNE and add the resulting coordinates to the dataframe
df = add_tsne_to_dataframe(
    df,
    perplexity=50,
)

# rename tsne columns to 2d_x, 2d_y
df = df.rename(columns={"tsne_1": "2d_x", "tsne_2": "2d_y"})
df.to_pickle(saved_path := "../tsne-final.pkl")
print(f"Saved to {saved_path}")
df

# Draw the t-SNE plot itself for our visualization

import matplotlib.pyplot as plt
import seaborn as sns


def draw_tsne_plot(df: pd.DataFrame, title: str = "t-SNE Plot"):
    """
    Draw a t-SNE plot for the given dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the t-SNE coordinates and labels
    title : str, optional
        Title of the plot, default is 't-SNE Plot'
    """
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
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()


draw_tsne_plot(df)

# %%
