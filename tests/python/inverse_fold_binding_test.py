from __future__ import annotations

import ast
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
for finder in list(sys.meta_path):
    if finder.__class__.__module__ == "_hikoboshi_editable_loader":
        sys.meta_path.remove(finder)

import hikoboshi as hiko

from helpers import require, write_simple_pdb


def npz_shape(path: Path) -> tuple[int, int, int]:
    with zipfile.ZipFile(path, "r") as archive:
        payload = archive.read("log_probs.npy")
    require(payload.startswith(b"\x93NUMPY"), "log-prob payload must be NPY")
    major = payload[6]
    require(major == 1, "expected NPY v1 payload")
    header_len = int.from_bytes(payload[8:10], "little")
    header = payload[10 : 10 + header_len].decode("ascii").strip()
    parsed = ast.literal_eval(header)
    shape = parsed["shape"]
    return tuple(int(value) for value in shape)


def valid_sequence(sequence: str, length: int) -> bool:
    return len(sequence) == length and all(
        residue in "ACDEFGHIKLMNPQRSTVWYX" for residue in sequence
    )


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    pdb = root / "input.pdb"
    logprobs = root / "lp.npz"
    write_simple_pdb(pdb, residue_count=3)

    first = hiko.inverse_fold(
        str(pdb),
        num_seqs=2,
        sampling_temp=0.1,
        seed=0,
    )
    second = hiko.inverse_fold(
        str(pdb),
        num_seqs=2,
        sampling_temp=0.1,
        seed=0,
        out_logprobs=str(logprobs),
    )

    require(len(first.sequences) == 2, "inverse_fold must return two sequences")
    require(
        [item.sequence for item in first.sequences]
        == [item.sequence for item in second.sequences],
        "inverse_fold must be deterministic for a fixed seed",
    )
    for item in first.sequences:
        require(valid_sequence(item.sequence, 3), "invalid designed sequence")
        require(item.decode_order == "random", "default decode_order must be random")
        require(item.seed >= 0, "sequence seed must be non-negative")

    require(second.logprobs.written, "out_logprobs must report a written artifact")
    require(second.logprobs.path == str(logprobs), "logprobs path mismatch")
    require(second.logprobs.shape == (2, 3, 21), "logprobs shape mismatch")
    require(logprobs.exists(), "logprobs artifact was not written")
    require(npz_shape(logprobs) == (2, 3, 21), "NPZ payload shape mismatch")

    n_to_c = hiko.inverse_fold.from_pdb(
        str(pdb),
        num_seqs=1,
        seed=0,
        decode_order="n_to_c",
    )
    require(n_to_c.sequences[0].decode_order == "n_to_c", "n_to_c decode order lost")
