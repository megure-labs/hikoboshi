from __future__ import annotations

import hikoboshi as hiko
from hikoboshi._arrays import load_core

from helpers import require


core = load_core()
Tensor = core.Tensor

require(getattr(hiko, "Tensor") is Tensor, "hikoboshi.Tensor must be registered")

tensor = Tensor([[1.0, 2.0], [3.5, 4.5]], dtype="float32")
require(tensor.shape == (2, 2), "Tensor shape property mismatch")
require(tensor.dtype == "float32", "Tensor dtype property mismatch")
require(tensor.device == "cpu", "Tensor device property mismatch")
require(len(tensor) == 2, "Tensor length mismatch")
require(tensor[0] == [1.0, 2.0], "Tensor row sequence mismatch")
require(tensor[-1] == [3.5, 4.5], "Tensor negative index mismatch")
require(tensor.tolist() == [[1.0, 2.0], [3.5, 4.5]], "Tensor tolist mismatch")

view = memoryview(tensor)
require(view.format == "f", "memoryview format mismatch")
require(view.shape == (2, 2), "memoryview shape mismatch")
require(view.strides == (8, 4), "memoryview strides mismatch")
require(not view.readonly, "Tensor buffer should be writable")
require(view.tolist() == [[1.0, 2.0], [3.5, 4.5]], "memoryview values mismatch")

flat = view.cast("B").cast("f")
flat[1] = 9.25
require(tensor.tolist() == [[1.0, 9.25], [3.5, 4.5]], "buffer write did not update Tensor")

reshaped = Tensor([1, 2, 3, 4], shape=(2, 2), dtype="int32")
require(reshaped.tolist() == [[1, 2], [3, 4]], "explicit shape reshape failed")
require(memoryview(reshaped).format == "i", "int32 buffer format mismatch")
