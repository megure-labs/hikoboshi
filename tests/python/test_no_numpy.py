from __future__ import annotations

import importlib.abc
import sys
import tempfile
from pathlib import Path

from helpers import matrix, require, write_simple_pdb


class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001, D102
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'")
        return None


for name in tuple(sys.modules):
    if name == "numpy" or name.startswith("numpy."):
        del sys.modules[name]
sys.meta_path.insert(0, BlockNumpy())

import hikoboshi as hiko  # noqa: E402
from hikoboshi._arrays import load_core  # noqa: E402


require(hiko.__version__ == "0.1.0", "package import must not require NumPy")
require(hiko.version_info()["product_name"] == "Hikoboshi", "version_info failed")

core = load_core()
require(hiko.Tensor is core.Tensor, "hikoboshi.Tensor alias mismatch")
tensor = hiko.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
require(tensor.shape == (2, 2), "Tensor basic construction failed")
require(memoryview(tensor).tolist() == [[1.0, 2.0], [3.0, 4.0]], "Tensor buffer failed")

query = matrix([[1.0, 0.0], [0.0, 1.0]])
target = matrix([[1.0, 0.0], [0.0, 1.0]])
pairwise = hiko.pairwise.from_embeddings(
    query,
    target,
    query_metadata={"residue_codes": "AC"},
    target_metadata={"residue_codes": "AC"},
)
require(pairwise.metrics.raw_sw_score == 2.0, "no-NumPy pairwise failed")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "query.pdb"
    write_simple_pdb(path)
    encoded = hiko.encode.from_structure(str(path), package="hikoboshi-mpnn-d64")

require(isinstance(encoded.embeddings, hiko.Tensor), "encode must return Tensor")
require(encoded.embeddings.shape == (2, 64), "encoded Tensor shape mismatch")
require(len(encoded.embeddings[0]) == 64, "encoded Tensor row shape mismatch")
