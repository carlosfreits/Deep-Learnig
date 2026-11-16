"""Converte um script no formato 'percent' (# %% / # %% [markdown]) em .ipynb."""
import re
import sys

import nbformat

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
parts = re.split(r"^# %%(.*)$", text, flags=re.M)
nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
for header, body in zip(parts[1::2], parts[2::2]):
    body = body.strip("\n")
    if "[markdown]" in header:
        md = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
        nb.cells.append(nbformat.v4.new_markdown_cell(md))
    elif body.strip():
        nb.cells.append(nbformat.v4.new_code_cell(body))
nbformat.write(nb, dst)
print(f"{dst}: {len(nb.cells)} células")
