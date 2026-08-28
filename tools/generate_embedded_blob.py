#!/usr/bin/env python3
"""Generate a C++ byte array from a checked binary model artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()

    payload = args.input.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != args.sha256:
        raise SystemExit(
            f"{args.input}: sha256 {actual} does not match {args.sha256}"
        )

    with args.output.open("w", encoding="ascii", newline="\n") as output:
        output.write(f'#include "{args.header}"\n\n')
        output.write(f"namespace {args.namespace} {{\n\n")
        output.write(
            "alignas(64) extern const std::uint8_t "
            "kSafetensorsBlob[kSafetensorsBlobLength] = {\n"
        )
        for offset in range(0, len(payload), 24):
            chunk = payload[offset : offset + 24]
            output.write("    ")
            output.write(", ".join(f"0x{value:02x}" for value in chunk))
            output.write(",\n")
        output.write("};\n\n")
        output.write(f"}}  // namespace {args.namespace}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
