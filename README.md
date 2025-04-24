# ECE6254 Project - Materials Embedding Visualization

This repository contains scripts for analyzing and visualizing embeddings from different materials science models using dimensionality reduction techniques like t-SNE and UMAP. Please visit our [demo page](http://ece6254.nima.sh/) for a live demonstration showcasting 20,000 generated crystal structures and their JMP embeddings.

## Project Structure

- `demo.py`: Main demonstration script for the project
- `tsne-final.pkl`: Pre-computed t-SNE/UMAP results saved as a pickle file
- `umap_scripts/`: Directory containing scripts for embedding conversion, dimensionality reduction, and visualization
  - See the [UMAP Scripts README](umap_scripts/README.md) for detailed documentation on these scripts
- `flowmm/`: Directory with FlowMM model implementation and documentation, see [FlowMM README](flowmm/README.md) for details
- `crystal-text-llm/`: Directory with Crystal Text LLM model implementation and documentation, see [Crystal Text LLM README](crystal-text-llm/README.md) for details

## Running the Demo

To visualize the pre-computed embeddings:

```bash
python demo.py
```

This will load the dimensionality reduction results from `tsne-final.pkl` and display the visualization.

## Prerequisites

For running the scripts in this project, you'll need the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn torch pytorch-lightning pymatgen torch-geometric torch-scatter tqdm rich
```

## UMAP Scripts

For detailed documentation on the UMAP scripts, including:
- Data conversion scripts
- Dimensionality reduction techniques (t-SNE and UMAP)
- Visualization tools
- How to run each script

Please see the [UMAP Scripts Documentation](umap_scripts/README.md).
