#!/usr/bin/env python3
"""Compare the native ProteinMPNN encoder layer to a tiny PyTorch EncLayer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("HIKOBOSHI_SOURCE_ROOT", SCRIPT_DIR.parents[2]))
DEFAULT_TOLERANCE = REPO_ROOT / "bench" / "numerical_tolerance.json"
HIDDEN = 128
NEIGHBORS = 48
RESIDUES = 5
MESSAGE_SCALE = 30.0
MASK64 = (1 << 64) - 1


def rerun_with_requested_python() -> None:
    target = os.environ.get("HIKOBOSHI_PYTORCH_PYTHON")
    if not target or os.environ.get("HIKOBOSHI_PYTORCH_RERUN"):
        return
    target_path = Path(target)
    if not target_path.exists():
        raise SystemExit(f"HIKOBOSHI_PYTORCH_PYTHON does not exist: {target}")
    if target_path.resolve() == Path(sys.executable).resolve():
        return
    env = os.environ.copy()
    env["HIKOBOSHI_PYTORCH_RERUN"] = "1"
    os.execve(str(target_path), [str(target_path), *sys.argv], env)


def require_python_deps():
    rerun_with_requested_python()
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        print(
            "SKIP: ProteinMPNN encoder layer parity requires NumPy and PyTorch. "
            "Set HIKOBOSHI_PYTORCH_PYTHON to a Python environment with both.",
            file=sys.stderr,
        )
        raise SystemExit(77) from exc
    return np, torch


def fail(message: str) -> None:
    raise SystemExit(message)


def load_tolerance(path: Path) -> float:
    if not path.is_file():
        return 1.5e-5
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("proteinmpnn_layer_abs")
    if not isinstance(value, (int, float)):
        fail(f"{path}: missing numeric proteinmpnn_layer_abs")
    value = float(value)
    if value > 1.5e-5:
        fail(f"{path}: proteinmpnn_layer_abs {value} exceeds 1.5e-5")
    return value


def splitmix(index: int, salt: int) -> int:
    x = 0x9E3779B97F4A7C15
    x = (x + (index + 1) * 0xBF58476D1CE4E5B9) & MASK64
    x = (x + (salt + 1) * 0x94D049BB133111EB) & MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return x & MASK64


def pseudo_value(index: int, salt: int, scale: float):
    np, _ = require_python_deps()
    bits = splitmix(index, salt) >> 40
    unit = np.float32(bits) * np.float32(1.0 / 8388608.0) - np.float32(1.0)
    return unit * np.float32(scale)


def pseudo_array(count: int, salt: int, scale: float):
    np, _ = require_python_deps()
    values = np.empty(count, dtype=np.float32)
    for index in range(count):
        values[index] = pseudo_value(index, salt, scale)
    return values


def norm_pair(salt: int):
    np, _ = require_python_deps()
    weight = np.empty(HIDDEN, dtype=np.float32)
    bias = np.empty(HIDDEN, dtype=np.float32)
    for index in range(HIDDEN):
        weight[index] = np.float32(0.9) + pseudo_value(index, salt, 0.08)
        bias[index] = pseudo_value(index, salt + 17, 0.03)
    return weight, bias


def fixture() -> dict[str, Any]:
    np, torch = require_python_deps()
    slots = RESIDUES * NEIGHBORS
    message_input = 3 * HIDDEN
    ffn_hidden = 4 * HIDDEN
    mask_v = np.asarray([1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    e_idx = np.full((RESIDUES, NEIGHBORS), -1, dtype=np.int32)
    mask_attend = np.zeros((RESIDUES, NEIGHBORS), dtype=np.float32)
    for residue in range(RESIDUES):
        for slot in range(NEIGHBORS):
            if slot < RESIDUES:
                neighbor = (residue + slot) % RESIDUES
                e_idx[residue, slot] = neighbor
                mask_attend[residue, slot] = mask_v[residue] * mask_v[neighbor]
    n1_w, n1_b = norm_pair(30)
    n2_w, n2_b = norm_pair(40)
    n3_w, n3_b = norm_pair(50)
    arrays = {
        "h_v": pseudo_array(RESIDUES * HIDDEN, 1, 0.35).reshape(RESIDUES, HIDDEN),
        "h_e": pseudo_array(slots * HIDDEN, 2, 0.25).reshape(
            RESIDUES, NEIGHBORS, HIDDEN
        ),
        "e_idx": e_idx,
        "mask_v": mask_v,
        "mask_attend": mask_attend,
        "W1_w": pseudo_array(HIDDEN * message_input, 10, 0.018).reshape(
            HIDDEN, message_input
        ),
        "W1_b": pseudo_array(HIDDEN, 11, 0.012),
        "W2_w": pseudo_array(HIDDEN * HIDDEN, 12, 0.015).reshape(HIDDEN, HIDDEN),
        "W2_b": pseudo_array(HIDDEN, 13, 0.010),
        "W3_w": pseudo_array(HIDDEN * HIDDEN, 14, 0.015).reshape(HIDDEN, HIDDEN),
        "W3_b": pseudo_array(HIDDEN, 15, 0.010),
        "W11_w": pseudo_array(HIDDEN * message_input, 16, 0.018).reshape(
            HIDDEN, message_input
        ),
        "W11_b": pseudo_array(HIDDEN, 17, 0.012),
        "W12_w": pseudo_array(HIDDEN * HIDDEN, 18, 0.015).reshape(HIDDEN, HIDDEN),
        "W12_b": pseudo_array(HIDDEN, 19, 0.010),
        "W13_w": pseudo_array(HIDDEN * HIDDEN, 20, 0.015).reshape(HIDDEN, HIDDEN),
        "W13_b": pseudo_array(HIDDEN, 21, 0.010),
        "dense_in_w": pseudo_array(ffn_hidden * HIDDEN, 22, 0.014).reshape(
            ffn_hidden, HIDDEN
        ),
        "dense_in_b": pseudo_array(ffn_hidden, 23, 0.010),
        "dense_out_w": pseudo_array(HIDDEN * ffn_hidden, 24, 0.014).reshape(
            HIDDEN, ffn_hidden
        ),
        "dense_out_b": pseudo_array(HIDDEN, 25, 0.010),
        "norm1_w": n1_w,
        "norm1_b": n1_b,
        "norm2_w": n2_w,
        "norm2_b": n2_b,
        "norm3_w": n3_w,
        "norm3_b": n3_b,
    }
    return {key: torch.from_numpy(value).to(torch.float32) for key, value in arrays.items()}


def linear(torch, x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


def gelu(torch, x):
    return torch.nn.functional.gelu(x, approximate="none")


def layer_norm(torch, x, weight, bias):
    return torch.nn.functional.layer_norm(x, (HIDDEN,), weight=weight, bias=bias, eps=1e-5)


def gather_nodes(torch, h_v, e_idx):
    valid = (e_idx >= 0) & (e_idx < h_v.shape[0])
    safe = torch.clamp(e_idx, min=0)
    gathered = h_v[safe.to(torch.long)]
    return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))


def cat_neighbors_nodes(torch, h_v, h_e, e_idx):
    query = h_v[:, None, :].expand(-1, h_e.shape[1], -1)
    gathered = gather_nodes(torch, h_v, e_idx)
    return torch.cat([query, h_e, gathered], dim=-1)


def run_reference():
    _, torch = require_python_deps()
    torch.set_num_threads(1)
    f = fixture()
    h_v = f["h_v"].clone()
    h_e = f["h_e"].clone()
    e_idx = f["e_idx"].to(torch.int64)

    h_ev = cat_neighbors_nodes(torch, h_v, h_e, e_idx)
    h_message = linear(torch, h_ev, f["W1_w"], f["W1_b"])
    h_message = linear(torch, gelu(torch, h_message), f["W2_w"], f["W2_b"])
    h_message = linear(torch, gelu(torch, h_message), f["W3_w"], f["W3_b"])
    h_message = f["mask_attend"].unsqueeze(-1) * h_message
    h_v = layer_norm(
        torch,
        h_v + torch.sum(h_message, dim=1) / MESSAGE_SCALE,
        f["norm1_w"],
        f["norm1_b"],
    )
    dense = linear(torch, h_v, f["dense_in_w"], f["dense_in_b"])
    dense = linear(torch, gelu(torch, dense), f["dense_out_w"], f["dense_out_b"])
    h_v = layer_norm(torch, h_v + dense, f["norm2_w"], f["norm2_b"])
    h_v = f["mask_v"].unsqueeze(-1) * h_v

    h_ev = cat_neighbors_nodes(torch, h_v, h_e, e_idx)
    h_message = linear(torch, h_ev, f["W11_w"], f["W11_b"])
    h_message = linear(torch, gelu(torch, h_message), f["W12_w"], f["W12_b"])
    h_message = linear(torch, gelu(torch, h_message), f["W13_w"], f["W13_b"])
    updated = layer_norm(torch, h_e + h_message, f["norm3_w"], f["norm3_b"])
    valid = ((e_idx >= 0) & (e_idx < RESIDUES)).unsqueeze(-1)
    h_e = torch.where(valid, updated, h_e)
    return {
        "h_v": h_v.detach().cpu().numpy(),
        "h_e": h_e.detach().cpu().numpy(),
        "e_idx": f["e_idx"].cpu().numpy(),
    }


def default_cpp_binary() -> Path:
    build_root = os.environ.get("HIKOBOSHI_BUILD_ROOT")
    if not build_root:
        fail("--cpp-binary or HIKOBOSHI_BUILD_ROOT is required")
    return Path(build_root) / "hikoboshi_proteinmpnn_encoder_layer_parity_test"


def run_cpp(binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), "--dump-json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def compare(native: dict[str, Any], reference: dict[str, Any], tolerance: float) -> None:
    np, _ = require_python_deps()
    h_v = np.asarray(native["h_v"], dtype=np.float32).reshape(RESIDUES, HIDDEN)
    h_e = np.asarray(native["h_e"], dtype=np.float32).reshape(
        RESIDUES, NEIGHBORS, HIDDEN
    )
    e_idx = np.asarray(native["e_idx"], dtype=np.int32).reshape(RESIDUES, NEIGHBORS)
    if not np.array_equal(e_idx, reference["e_idx"]):
        fail("native E_idx output does not retain input neighbor indices")
    node_max = float(np.max(np.abs(h_v - reference["h_v"])))
    edge_max = float(np.max(np.abs(h_e - reference["h_e"])))
    if node_max > tolerance or edge_max > tolerance:
        fail(
            "ProteinMPNN encoder layer parity failed: "
            f"h_V max_abs={node_max:.9g}, h_E max_abs={edge_max:.9g}, "
            f"tolerance={tolerance:.9g}"
        )
    print(
        "proteinmpnn_encoder_layer_parity: pass "
        f"h_V max_abs={node_max:.9g} h_E max_abs={edge_max:.9g} "
        f"tolerance={tolerance:.9g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-binary", type=Path, default=None)
    parser.add_argument("--tolerance", type=Path, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    binary = args.cpp_binary if args.cpp_binary is not None else default_cpp_binary()
    tolerance = load_tolerance(args.tolerance)
    native = run_cpp(binary)
    reference = run_reference()
    compare(native, reference, tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
