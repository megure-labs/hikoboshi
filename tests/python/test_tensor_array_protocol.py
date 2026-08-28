from __future__ import annotations

import sys

from hikoboshi._arrays import load_core

from helpers import require, skip


try:
    import numpy as np
except ModuleNotFoundError:
    skip("NumPy is not installed; skipping hikoboshi.Tensor NumPy buffer interop")


Tensor = load_core().Tensor
source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
tensor = Tensor(source)

array = np.asarray(tensor)
require(array.dtype == np.float32, "NumPy view dtype mismatch")
require(array.shape == (2, 2), "NumPy view shape mismatch")
require(array.tolist() == [[1.0, 2.0], [3.0, 4.0]], "NumPy view values mismatch")

array[1, 0] = np.float32(7.5)
require(tensor.tolist() == [[1.0, 2.0], [7.5, 4.0]], "NumPy view must share Tensor memory")

roundtrip = Tensor(array)
require(roundtrip.tolist() == [[1.0, 2.0], [7.5, 4.0]], "NumPy input copy mismatch")

sys.exit(0)
