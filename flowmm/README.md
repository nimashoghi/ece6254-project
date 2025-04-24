# FlowMM Setup, Training, and Usage

1. Set up the [FlowMM repository](https://github.com/facebookresearch/flowmm/) according to the instructions in the repository.
2. For training, use the commands below:
    - For de-novo generation:
        ```bash
        python scripts_model/run.py data=mp_20 model=abits_params
        ```
    - For crystal structure prediction:
        ```bash
        python scripts_model/run.py data=mp_20 model=null_params
        ```
3. Once trained, you can use the following scripts:
    - For de-novo generation:
        ```bash
        ckpt=path/to/your/checkpoint
        subdir=your_subdir
        slope=5.0
        batch_size=32
        # export CUDA_VISIBLE_DEVICES=... # set this if you want to use a specific GPU
        python scripts_model/evaluate.py generate ${ckpt} --subdir ${subdir} --inference_anneal_slope ${slope} --batch_size ${batch_size} && \
        python scripts_model/evaluate.py consolidate ${ckpt} --subdir ${subdir} && \
        python scripts_model/evaluate.py old_eval_metrics ${ckpt} --subdir ${subdir} --stage test && \
        python scripts_model/evaluate.py lattice_metrics ${ckpt} --subdir ${subdir} --stage test
        ```
    - For crystal structure prediction:
        ```bash
        ckpt=path/to/your/checkpoint
        subdir=your_subdir
        slope=10.0
        batch_size=32
        # export CUDA_VISIBLE_DEVICES=... # set this if you want to use a specific GPU
        python scripts_model/evaluate.py reconstruct ${ckpt} --subdir ${subdir} --inference_anneal_slope ${slope} --stage test --batch_size ${batch_size} && \
        python scripts_model/evaluate.py consolidate ${ckpt} --subdir ${subdir} && \
        python scripts_model/evaluate.py old_eval_metrics ${ckpt} --subdir ${subdir} --stage test && \
        python scripts_model/evaluate.py lattice_metrics ${ckpt} --subdir ${subdir} --stage test
        ```

## FlowMM Data Processing

The `4-12-dump.py` script is used to process the outputs from FlowMM model runs and prepare them for visualization:

### What it does
1. Loads generated crystal structures from a FlowMM checkpoint
2. Processes the generated crystals (filtering invalid structures)
3. Converts them to PyMatGen structures in a pandas DataFrame
4. Saves the processed data as a pickle file (`mp20_test.pkl`) for use with the UMAP visualization scripts

### How to run it

1. Make sure you have a trained FlowMM model with generated structures.
2. Update the `base_path` and `ckpt_path` variables in the script to point to your generated structures and checkpoint
3. Run the script:

```bash
cd flowmm
python 4-12-dump.py
```

4. The script will generate `mp20_test.pkl` which can then be used by the UMAP scripts for embedding generation

### Data Flow

1. FlowMM model generates crystal structures → Saved as consolidated_generate.pt
2. 4-12-dump.py processes these structures → Saved as mp20_test.pkl
3. umap_scripts/4-12-convert-flowmm.py converts these to embeddings → Saved as flowmm_mp20_test_with_embeddings.pkl
4. UMAP/t-SNE scripts use these embeddings for visualization
