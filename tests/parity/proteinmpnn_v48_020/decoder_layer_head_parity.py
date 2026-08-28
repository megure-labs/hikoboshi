#!/usr/bin/env python3
"""Synthetic ProteinMPNN v_48_020 decoder-head logits parity."""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    print(
        "SKIP: decoder-layer parity requires NumPy.",
        file=__import__("sys").stderr,
    )
    raise SystemExit(77) from exc

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised on lean dev envs.
    torch = None
    F = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("HIKOBOSHI_SOURCE_ROOT", SCRIPT_DIR.parents[2]))
TOLERANCE_PATH = REPO_ROOT / "bench" / "numerical_tolerance.json"
DEFAULT_TOLERANCE = 1.5e-5


def load_tolerance() -> float:
    if not TOLERANCE_PATH.is_file():
        return DEFAULT_TOLERANCE
    payload = json.loads(TOLERANCE_PATH.read_text(encoding="ascii"))
    value = payload.get("proteinmpnn_layer_abs", DEFAULT_TOLERANCE)
    if not isinstance(value, int | float):
        raise SystemExit(f"{TOLERANCE_PATH}: proteinmpnn_layer_abs is not numeric")
    return min(float(value), DEFAULT_TOLERANCE)


def main() -> None:
    rng = np.random.default_rng(0x48020DEC)
    residue_count = 7
    hidden = 128
    vocab = 21
    h_v = rng.normal(0.0, 0.25, size=(residue_count, hidden)).astype(np.float32)
    weight = rng.normal(0.0, 0.08, size=(vocab, hidden)).astype(np.float32)
    bias = rng.normal(0.0, 0.03, size=(vocab,)).astype(np.float32)

    candidate_logits = h_v @ weight.T + bias
    if torch is not None:
        linear = torch.nn.Linear(hidden, vocab, bias=True)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(weight))
            linear.bias.copy_(torch.from_numpy(bias))
        reference_logits = linear(torch.from_numpy(h_v)).detach().numpy()
    else:
        reference_logits = candidate_logits.copy()

    max_abs = float(np.max(np.abs(candidate_logits - reference_logits)))
    tolerance = load_tolerance()
    if max_abs > tolerance:
        raise SystemExit(
            f"decoder_layer_head_parity: logits max_abs={max_abs:.8g} "
            f"tolerance={tolerance:.8g}"
        )

    if torch is not None:
        log_probs = F.log_softmax(torch.from_numpy(candidate_logits), dim=-1).numpy()
    else:
        row_max = np.max(candidate_logits, axis=-1, keepdims=True)
        log_z = row_max + np.log(
            np.sum(np.exp(candidate_logits - row_max), axis=-1, keepdims=True)
        )
        log_probs = candidate_logits - log_z
    if float(np.max(np.abs(candidate_logits - log_probs))) <= tolerance:
        raise SystemExit("decoder_layer_head_parity: head emitted log_softmax")

    print(
        "decoder_layer_head_parity: "
        f"logits max_abs={max_abs:.8g} tolerance={tolerance:.8g}"
    )


if __name__ == "__main__":
    main()
