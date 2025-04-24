# %%
from __future__ import annotations

from pathlib import Path

import nshutils as nu
import numpy as np
import rich
import torch
import torch.utils._pytree as tree

nu.pretty()

# %%
import datasets

dataset = datasets.load_dataset("nimashoghi/mp20", split="all")
assert isinstance(dataset, datasets.Dataset)

# Subsample a random 10k
dataset = dataset.shuffle(seed=42).select(range(10000))

dataset

# %%
from pymatgen.core import Structure
from tqdm.auto import tqdm

structures_list = []
for data_dict in tqdm(dataset, total=len(dataset)):
    cif = data_dict["cif"]
    structure = Structure.from_str(cif, fmt="cif")
    structures_list.append(structure)

print(len(structures_list))


# %%
import torch
from jmp.models.gemnet.graph import GraphComputer
from lightning.fabric.utilities.apply_func import move_data_to_device

from mlff_uq.experiments import jmp_graph_property as M
from mlff_uq.paths import nai_paths as P
from mlff_uq.utils.jmp_loader_cached import jmp_from_pretrained_ckpt_cached

device = torch.device("cuda:0")
# device = torch.device("cpu")

(no_grad_ctx := torch.no_grad()).__enter__()
(inference_mode_ctx := torch.inference_mode()).__enter__()

ckpt_path = P.ckpts_dir / "jmp-s.pt"
model = jmp_from_pretrained_ckpt_cached(ckpt_path)
model = model.to(device).eval()
for param in model.parameters():
    param.requires_grad = False


def jmp_graph_computer():
    graph_computer = M.JMPGraphComputerConfig.draft()
    graph_computer.pbc = True
    graph_computer.cutoffs = M.CutoffsConfig.from_constant(12.0)
    graph_computer.max_neighbors = M.MaxNeighborsConfig.from_goc_base_proportions(30)

    graph_computer = GraphComputer(
        graph_computer._to_jmp_graph_computer_config(), model.hparams
    )
    return graph_computer


graph_computer = jmp_graph_computer()

print(model, graph_computer)

# %%
from pymatgen.core import Structure
from torch_geometric.data import Batch, Data
from torch_scatter import scatter
from tqdm.auto import trange


def create_data(structure: Structure):
    ddict: dict[str, torch.Tensor] = {}

    ddict["pos"] = torch.tensor(structure.cart_coords, dtype=torch.float32)
    ddict["atomic_numbers"] = torch.tensor(structure.atomic_numbers, dtype=torch.long)
    ddict["cell"] = torch.tensor(structure.lattice.matrix, dtype=torch.float32).reshape(
        1, 3, 3
    )
    ddict["natoms"] = torch.tensor([structure.num_sites], dtype=torch.long)
    ddict["tags"] = torch.full_like(ddict["atomic_numbers"], 2, dtype=torch.long)
    ddict["pbc"] = torch.tensor([True, True, True], dtype=torch.bool).reshape(1, 3)

    return Data.from_dict(ddict)


def collate_fn(data_list):
    return Batch.from_data_list(data_list)


def create_batch(structures: list[Structure]):
    data_list = [create_data(s) for s in structures]
    batch = collate_fn(data_list)
    batch = move_data_to_device(batch, device)
    return batch


def process_batch(batch: Batch):
    batch = graph_computer(batch)
    model_output = model(batch)
    node_embeddings = model_output["energy"]
    # ^ (natoms, d)
    node_embeddings = scatter(
        node_embeddings,
        batch.batch,
        dim=0,
        dim_size=batch.num_graphs,
        reduce="mean",
    )
    # ^ (ngraphs, d)
    return node_embeddings.float().cpu().numpy()


def process_structures_in_batches(
    structures: list[Structure],
    batch_size: int = 32,
    fault_tolerant: bool = True,
):
    """
    Process a list of pymatgen Structure objects in batches and return aggregated node embeddings.

    Parameters
    ----------
    structures : list[Structure]
        List of pymatgen Structure objects to process.
    batch_size : int, optional
        Number of structures to process per batch. Adjust this based on available memory. Default is 32.
    fault_tolerant : bool, optional
        If True, errors in individual batches will be ignored and processing will continue. Default is False.

    Returns
    -------
    numpy.ndarray
        Combined node embeddings from all batches with shape (n_structures, embedding_dimension).
    """
    embeddings_list = []
    skipped_indices = []

    # Process the list in batches
    for start in trange(0, len(structures), batch_size):
        # Slice a batch of structures
        batch_structures = structures[start : start + batch_size]

        try:
            # Create a torch_geometric batch from the current list of structures
            batch = create_batch(batch_structures)

            # Process the batch: run through the graph computer, model, and reduce with scatter
            embeddings = process_batch(batch)
            embeddings_list.append(embeddings)
        except Exception as e:
            if fault_tolerant:
                # In fault tolerant mode, log the error and continue
                print(f"Error processing batch {start}-{start + batch_size - 1}: {e}")
                skipped_indices.extend(
                    range(start, min(start + batch_size, len(structures)))
                )
            else:
                # Re-raise the exception if not in fault tolerant mode
                raise

    if skipped_indices and fault_tolerant:
        print(f"Skipped {len(skipped_indices)} structures due to errors")

    # Concatenate all the embeddings vertically into one array
    return np.vstack(embeddings_list) if embeddings_list else np.array([])


# %%
import pandas as pd

df = pd.DataFrame({"structure": structures_list})
df["dataset"] = "mp20"
df["subset"] = "test"
df["model"] = "ground_truth"
df["dataset_full_name"] = "Ground Truth - MP20"
df

# %%
all_embeddings = process_structures_in_batches(
    df["structure"].tolist(),
    batch_size=32,
    fault_tolerant=True,
)
print(all_embeddings.shape)


# %%
df_copy = df.copy()
# add embeddings
df_copy["embeddings"] = list(all_embeddings)
df_copy["model"] = "ground_truth_with_embeddings"
df_copy.to_pickle("./ground_truth_with_embeddings.pkl")
df_copy

# %%
