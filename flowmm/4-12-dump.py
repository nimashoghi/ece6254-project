# %%

from __future__ import annotations

from pathlib import Path

from IPython.core.getipython import get_ipython

if (ipython := get_ipython()) is not None:
    ipython.run_line_magic("load_ext", "dotenv")
    ipython.run_line_magic("dotenv", "")
    print("Loaded .env file")

    import os

    print(os.environ["PROJECT_ROOT"])

import nshutils as nu
import numpy as np
import rich
import torch
import torch.utils._pytree as tree

nu.pretty()
# %% SET THESE CORRECTLY
base_path = Path(
    "/home/nimashoghi/repositories/flowmm/runs/trash/2025-04-01/23-02-19/abits_params-rfm_cspnet-k755y73e/every_n_epochs/results_4-3"
)
assert base_path.exists(), f"{base_path} does not exist"
assert base_path.is_dir(), f"{base_path} is not a directory"
# !ll {base_path}

ckpt_path = Path(
    "/home/nimashoghi/repositories/flowmm/runs/trash/2025-04-01/23-02-19/abits_params-rfm_cspnet-k755y73e/every_n_epochs/epoch=1999-step=212000.ckpt"
)
assert ckpt_path.exists(), f"{ckpt_path} does not exist"


# %%
consolidated_generate_path = base_path / "consolidated_generate.pt"
assert consolidated_generate_path.exists(), (
    f"{consolidated_generate_path} does not exist"
)
consolidated_generate = torch.load(consolidated_generate_path, map_location="cpu")

# %%
from typing import Any

import lovely_numpy as ln
import lovely_tensors as lt


def _info(v: Any):
    match v:
        case list():
            if not v:
                return "[<empty>]"
            return f"list@{len(v)}[{_info(v[0])}]"
        case torch.Tensor():
            return str(lt.lovely(v))
        case np.ndarray():
            return str(ln.lovely(v))
        case dict():
            return {k: _info(v) for k, v in v.items()}
        case _:
            return str(v)


def pprint(v: Any):
    rich.print(_info(v))


pprint(consolidated_generate)

# %%
from flowmm.model.eval_utils import load_cfg

stage = "test"


def load_config(ckpt: Path):
    cfg = load_cfg(ckpt)
    return cfg


cfg = load_config(ckpt_path)
rich.print(cfg)

gt_dataset_path = cfg.data.datamodule.datasets[stage][0].save_path
eval_model_name = cfg.data.eval_model_name
rich.print(
    {
        "gt_dataset_path": gt_dataset_path,
        "eval_model_name": eval_model_name,
    }
)

# %%
from dataclasses import dataclass

from flowmm.old_eval.core import Crystal, get_Crystal_obj_lists
from flowmm.old_eval.generation_metrics import InvalidCrystal


@dataclass
class ProcessedCrystals:
    generated_crystals: list[Crystal]
    ground_truth_crystals: list[Crystal]
    n_samples: int
    eval_model_name: str


def process(
    consolidated_generation_path: Path,
    ground_truth_path: Path,
    n_subsamples: int = 1_000,
):
    gen_crys, gt_crys, _ = get_Crystal_obj_lists(
        consolidated_generation_path,
        multi_eval=False,
        ground_truth_path=ground_truth_path,
    )

    # here we drop absurd crystals and crystals that didn't construct
    gen_crys_out = []
    for gc in gen_crys:
        if not (gc.lengths == 100).all() and gc.constructed:
            gen_crys_out.append(gc)
        else:
            gen_crys_out.append(InvalidCrystal())

    return ProcessedCrystals(
        generated_crystals=gen_crys_out,
        ground_truth_crystals=gt_crys,
        n_samples=n_subsamples,
        eval_model_name=eval_model_name,
    )


processed_crystals = process(
    consolidated_generation_path=consolidated_generate_path,
    ground_truth_path=gt_dataset_path,
    n_subsamples=1_000,
)
# %%
torch.save(processed_crystals, "processed_crystals.pt")

# %%
from flowmm.old_eval.core import Crystal


def process_crystal(crystal: Crystal):
    # Get pytmatgen structure
    structure = crystal.to_pymatgen_structure()
    return structure


process_crystal(processed_crystals.ground_truth_crystals[0])

# %%
import pandas as pd
from tqdm.auto import tqdm

assert len(processed_crystals.generated_crystals) == 10_000

df = pd.DataFrame(
    {
        "structure": [
            process_crystal(crystal)
            for crystal in tqdm(processed_crystals.generated_crystals)
        ]
    }
)
df["dataset"] = "mp20"
df["subset"] = "test"
df["model"] = "flowmm"
df["dataset_full_name"] = "FlowMM - MP20 Test"
df.to_pickle("./mp20_test.pkl")

# %%
