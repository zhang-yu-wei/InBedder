import os
from tqdm import tqdm

embed_files = []
def find_file(substring):
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(substring):
                embed_files.append(os.path.join(root, f))

find_file(".safetensors")

print(embed_files)
# breakpoint()
excluded_models = []

for f in tqdm(embed_files):
    excluded = False
    for m in excluded_models:
        if m in f:
            excluded = True
            break
    if excluded:
        continue
    if os.path.exists(f):
        os.remove(f)