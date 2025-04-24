# %%
import pandas as pd

df = pd.read_csv("./saved_samples/llama-2-7B_0.7_0.7.csv")
df

# %%
from llama_sample import parse_fn_to_struct


def tryparse(s: str):
    s = s[
        len(
            "Below is a description of a bulk material. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:"
        ) :
    ].strip()
    return parse_fn_to_struct(s)


generated = df["gen_str"].tolist()
print(len(generated))

# %%
from tqdm.auto import tqdm

count = len(generated)
pbar = tqdm(total=count)
num_generated = 0
structures = []
while num_generated < count:
    try:
        s = generated.pop()
        structures.append(tryparse(s))
    except Exception as e:
        pbar.write(f"Error: {e}")
    else:
        pbar.update(1)
        num_generated += 1

# %%
df = pd.DataFrame({"structure": [s for s in structures]})
df["dataset"] = "mp20"
df["subset"] = "test"
df["model"] = "crystaltextllm"
df["dataset_full_name"] = "CrystalTextLLM - MP20 Test"
df.to_pickle("./mp20_test.pkl")

# %%
