from __future__ import annotations

from hikoboshi._arrays import load_core

from helpers import require, skip


Tensor = load_core().Tensor
tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

require(tensor.__dlpack_device__() == (1, 0), "DLPack device tuple mismatch")
capsule = tensor.__dlpack__()
require(type(capsule).__name__ == "PyCapsule", "DLPack export must return a capsule")
del capsule

try:
    import numpy as np
except ModuleNotFoundError:
    skip("NumPy is not installed; skipping hikoboshi.Tensor NumPy DLPack interop")


array = np.from_dlpack(tensor)
require(array.dtype == np.float32, "DLPack NumPy dtype mismatch")
require(array.shape == (2, 2), "DLPack NumPy shape mismatch")
require(array.tolist() == [[1.0, 2.0], [3.0, 4.0]], "DLPack NumPy values mismatch")
device = getattr(array, "device", "cpu")
require(str(device).lower() == "cpu", "DLPack NumPy device mismatch")
