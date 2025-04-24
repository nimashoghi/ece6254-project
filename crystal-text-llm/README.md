# Crystal Text LLM Setup, Training, and Usage

1. Set up the [Crystal Text LLM repository](https://github.com/facebookresearch/crystal-text-llm) according to the instructions in the repository.
2. For training, use the command below:
    ```bash
    python llama_finetune.py \
        --run-name="7b-paper-replica" \
        --model_name="7b" \
        --num-epochs=65 \
        --batch-size=2 \
        --grad-accum=1 \
        --lr=5e-4 \
        --lr-scheduler="cosine" \
        --num-warmup-steps=100 \
        --weight-decay=0.0 \
        --lora-rank=8 \
        --lora-alpha=32 \
        --lora-dropout=0.05 \
        --fp8 \
        --eval-freq=1000 \
        --save-freq=500
    ```
3. Once trained, you can use the following scripts:
    - For de-novo generation:
        ```bash
        python llama_sample.py --model_name 7b --model_path=path/to/your/trained/model --out_path=llm_samples_denovo.csv
        ```
    - For crystal structure prediction:
        ```bash
        python llama_sample.py --model_name 7b --model_path=path/to/your/trained/model --conditions_file ./data/with_tags/test.csv --out_path=llm_samples_csp.csv
        ```

## Crystal Text LLM Data Processing

The `4-12-dump.py` script is used to process the outputs from Crystal Text LLM model runs and prepare them for visualization:

### What it does
1. Loads generated crystal structures from a Crystal Text LLM checkpoint
2. Processes the generated crystals (filtering invalid structures)
3. Converts them to PyMatGen structures in a pandas DataFrame
4. Saves the processed data as a pickle file (`mp20_test.pkl`) for use with the UMAP visualization scripts

### How to run it

1. Make sure you have a trained Crystal Text LLM model with generated structures.
2. Update the `base_path` and `ckpt_path` variables in the script to point to your generated structures and checkpoint
3. Run the script:

```bash
cd crystal Text LLM
python 4-12-dump.py
```

4. The script will generate `mp20_test.pkl` which can then be used by the UMAP scripts for embedding generation
