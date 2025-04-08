from collections import Counter
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov
)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


def safe_crystal(crys_dict, idx=None):
    try:
        return Crystal(crys_dict)
    except Exception as e:
        print(f"[CRYSTAL INIT ERROR] Index {idx}: {e}")
        return None


def get_file_paths(root_path, task, label='', suffix='pt'):
    if label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    return os.path.join(root_path, out_name)


def get_crystal_array_list(file_path, batch_idx=0, limit=3000):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['lengths'][batch_idx],
        data['angles'][batch_idx],
        data['num_atoms'][batch_idx])

    crys_array_list = crys_array_list[:limit]  # Limit here

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def main(args):
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'gen' in args.tasks:
        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path, limit=args.limit)
        print(f"[INFO] Evaluating {len(crys_array_list)} crystals...")
        gen_crys = p_map(lambda i: safe_crystal(crys_array_list[i], i), range(len(crys_array_list)))
        gen_crys = [c for c in gen_crys if c is not None]

        _, true_crystal_array_list = get_crystal_array_list(recon_file_path, limit=args.limit)
        gt_crys = p_map(lambda x: safe_crystal(x), true_crystal_array_list)
        gt_crys = [c for c in gt_crys if c is not None]

        from compute_metrics import GenEval
        gen_evaluator = GenEval(gen_crys, gt_crys, n_samples=min(args.limit, 1000), eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    print("[RESULT]", all_metrics)

    metrics_out_file = f"eval_metrics_{args.label}.json" if args.label else "eval_metrics.json"
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)
    with open(metrics_out_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default = "/home/jamshid/workspace/nima/cdvae/hydra/singlerun/2025-04-02/mp_20/eval_gen.pt" ,required=False)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['gen'], required=False)
    parser.add_argument('--limit', type=int, default=3000, required=False)
    args = parser.parse_args()
    main(args)