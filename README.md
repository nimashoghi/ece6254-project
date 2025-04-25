# ECE6254 Project - Materials Embedding Visualization

This repository contains scripts for analyzing and visualizing embeddings from different materials science models using dimensionality reduction techniques like t-SNE and UMAP. Please visit our [demo page](http://ece6254.nima.sh/) for a live demonstration showcasting 20,000 generated crystal structures and their JMP embeddings.

## Project Structure

- `demo.py`: Main demonstration script for the project. This is the script behind the demo page.
- `tsne-final.pkl`: Pre-computed t-SNE/UMAP results saved as a pickle file
- `umap_scripts/`: Directory containing scripts for embedding conversion, dimensionality reduction, and visualization. See the [UMAP Scripts README](umap_scripts/README.md) for detailed documentation on these scripts
- `flowmm/`: Directory with FlowMM model implementation and documentation, see [FlowMM README](flowmm/README.md) for details
- `crystal-text-llm/`: Directory with Crystal Text LLM model implementation and documentation, see [Crystal Text LLM README](crystal-text-llm/README.md) for details
- `cdvae/`: Directory with CDVAE model implementation and documentation, see [CDVAE README](cdvae/README.md) for details
