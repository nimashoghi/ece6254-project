# UMAP Scripts Documentation

This directory contains Python scripts for processing and visualizing material embeddings using dimensionality reduction techniques like t-SNE and UMAP.

## Overview

These scripts convert raw model outputs into embeddings and then apply dimensionality reduction techniques to visualize the high-dimensional embeddings in 2D space.

## Data Conversion Scripts

These scripts convert raw model outputs into embeddings that can be visualized:

- **4-12-convert-cdvae.py**: Converts CDVAE model outputs to embeddings
  - Loads raw CDVAE data from `data/cdvae_mp20_test.pkl`
  - Uses a pre-trained model to generate embeddings for crystal structures
  - Saves the results to `data/cdvae_mp20_test_with_embeddings.pkl`

- **4-12-convert-crystal-text-llm.py**: Converts Crystal Text LLM model outputs to embeddings
  - Loads raw data from `data/crystal_text_llm_mp20_test.pkl`
  - Processes structures to generate embeddings
  - Saves results to `data/crystal_text_llm_mp20_test_with_embeddings.pkl`

- **4-12-convert-flowmm.py**: Converts FlowMM model outputs to embeddings
  - Loads raw data from `data/flowmm_mp20_test.pkl`
  - Processes structures to generate embeddings
  - Saves results to `data/flowmm_mp20_test_with_embeddings.pkl`

- **4-13-convert-mp20.py**: Converts MP20 (Materials Project) data to embeddings
  - Processes MP20 dataset to extract embeddings

## Dimensionality Reduction Scripts

These scripts apply dimensionality reduction techniques to visualize the high-dimensional embeddings:

- **4-12-tsne.py**: Applies t-SNE dimensionality reduction to the embeddings
  - Loads embeddings from all models and ground truth data
  - Applies t-SNE with cosine distance metric
  - Saves the results to `../tsne-final.pkl`
  - Includes visualization functionality to create t-SNE plots

- **4-13-umap.py**: Applies UMAP dimensionality reduction to the embeddings
  - Similar to the t-SNE script but uses UMAP algorithm instead
  - Loads embeddings from all models and ground truth data
  - Applies UMAP with cosine distance metric
  - Saves the results to `../tsne-final.pkl` (note the filename is the same despite using UMAP)
  - Includes visualization functionality

## Visualization Scripts

- **fig.py**: Creates 3D visualizations of crystal unit cells
  - Generates plots showing unit cells and periodic tiling of crystals
  - Saves visualization as `unit_cell.png`

- **umap-for-paper.py**: Creates publication-quality UMAP visualizations
  - Uses the pre-computed dimensionality reduction results
  - Generates high-quality plots suitable for inclusion in research papers

## How to Use the Scripts

### Prerequisites

Before running the scripts, ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn torch pytorch-lightning pymatgen torch-geometric torch-scatter tqdm rich
```

### Data Conversion

To convert model outputs to embeddings:

1. Ensure you have the raw data files in the `data/` directory:
   - `cdvae_mp20_test.pkl`
   - `crystal_text_llm_mp20_test.pkl`
   - `flowmm_mp20_test.pkl`

2. Run the conversion scripts in order:
   ```bash
   python 4-12-convert-cdvae.py
   python 4-12-convert-crystal-text-llm.py
   python 4-12-convert-flowmm.py
   ```

### Dimensionality Reduction and Visualization

After generating embeddings, run either t-SNE or UMAP:

```bash
# For t-SNE visualization
python 4-12-tsne.py

# For UMAP visualization
python 4-13-umap.py
```

These scripts will:
1. Load embeddings from all models
2. Apply dimensionality reduction
3. Save the results to `../tsne-final.pkl`
4. Display the visualization plot

### Creating Publication Figures

To create publication-quality figures:

```bash
# For unit cell visualization
python fig.py

# For high-quality UMAP plots for publications
python umap-for-paper.py
```

## Data Files

The `data/` directory contains:

- Raw model outputs:
  - `cdvae_mp20_test.pkl`
  - `crystal_text_llm_mp20_test.pkl`
  - `flowmm_mp20_test.pkl`

- Processed embeddings:
  - `cdvae_mp20_test_with_embeddings.pkl`
  - `crystal_text_llm_mp20_test_with_embeddings.pkl`
  - `flowmm_mp20_test_with_embeddings.pkl`
  - `ground_truth_with_embeddings.pkl`
