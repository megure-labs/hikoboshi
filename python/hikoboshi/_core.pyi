from __future__ import annotations

from typing import Any, Sequence, TypedDict

ALIGNMENT_GAP_SENTINEL: int
__version__: str


class HikoboshiError(RuntimeError):
    code: str
    status_code: str
    detail: str


class InvalidArgumentError(HikoboshiError): ...
class FailedPreconditionError(HikoboshiError): ...
class UnavailableError(HikoboshiError): ...
class UnimplementedError(HikoboshiError): ...
class InternalError(HikoboshiError): ...


class Tensor:
    shape: tuple[int, ...]
    dtype: str
    device: str

    def __init__(
        self,
        data: Any,
        shape: Sequence[int] | None = None,
        dtype: str | None = None,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...
    def __dlpack__(
        self,
        stream: Any = None,
        max_version: Any = None,
        dl_device: Any = None,
        copy: Any = None,
    ) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def tolist(self) -> Any: ...


class BackendAvailability(TypedDict):
    compiled: bool
    runtime_available: bool
    reason: str


class GpuBackendAvailability(BackendAvailability):
    devices: list[str]


class CpuBackendCapabilities(TypedDict):
    scalar: BackendAvailability
    sse4: BackendAvailability
    avx2: BackendAvailability
    avx512: BackendAvailability
    neon: BackendAvailability
    sve: BackendAvailability


class GpuBackendCapabilities(TypedDict):
    cuda: GpuBackendAvailability
    hip: GpuBackendAvailability
    metal: GpuBackendAvailability
    vulkan: GpuBackendAvailability
    opencl: GpuBackendAvailability


class PipelineCapabilities(TypedDict):
    pairwise_alignment: bool
    symmetric_all_vs_all: bool
    structure_inputs: bool
    embedding_inputs: bool


class BackendCapabilities(TypedDict):
    cpu: CpuBackendCapabilities
    gpu: GpuBackendCapabilities
    pipeline: PipelineCapabilities
    default_backend: str


class MetricValuePayload(TypedDict):
    value: float
    valid: bool
    reason: str


class PairwiseMetricsPayload(TypedDict, total=False):
    raw_sw_score: float
    soft_sw_score: MetricValuePayload


class InverseFoldSequencePayload(TypedDict):
    sequence: str
    score: float
    recovery_vs_native: MetricValuePayload
    seed: int
    decode_order: str


class InverseFoldLogProbsPayload(TypedDict):
    path: str
    shape: tuple[int, int, int]
    written: bool


class InverseFoldResultPayload(TypedDict):
    sequences: list[InverseFoldSequencePayload]
    logprobs: InverseFoldLogProbsPayload


def version_info() -> dict[str, Any]: ...
def backend_capabilities() -> BackendCapabilities: ...
def default_weights_info() -> dict[str, Any]: ...
def compiled_models_info() -> dict[str, Any]: ...
def pairwise_from_embeddings(
    query_embeddings: Any,
    target_embeddings: Any,
    query_metadata: dict[str, Any] | None = None,
    target_metadata: dict[str, Any] | None = None,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def all_vs_all_from_embeddings(
    embeddings: Sequence[Any],
    metadata: Sequence[dict[str, Any] | None] | None = None,
    include_self: bool = False,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def load_structure_metadata(
    path: str,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
) -> dict[str, Any]: ...
def encode_from_structure(
    structure: Any,
    *,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
    package: str | None = None,
) -> dict[str, Any]: ...
def encode_from_coords(
    coords: Any,
    *,
    package: str | None = None,
) -> dict[str, Any]: ...
def inverse_fold_from_structure(
    structure: Any,
    *,
    num_seqs: int = 1,
    sampling_temp: float = 0.1,
    seed: int = 0,
    decode_order: str = "random",
    package: str | None = None,
    backbone_noise: float = 0.0,
    out_logprobs: str | None = None,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
) -> InverseFoldResultPayload: ...
def pairwise_from_structure(
    query: Any,
    target: Any,
    *,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
    package: str | None = None,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def pairwise_from_coords(
    query_coords: Any,
    target_coords: Any,
    *,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def pair_list_from_sequence(
    pairs: Sequence[tuple[str, str]],
    fasta_path: str,
    *,
    package: str | None = None,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> list[dict[str, Any]]: ...
def pair_list_from_structure(
    pairs: Sequence[tuple[str, str]],
    pdb_dir: str,
    *,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> list[dict[str, Any]]: ...
def pair_list_from_coords(
    pairs: Sequence[tuple[str, str]],
    coords: Sequence[Any],
    *,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> list[dict[str, Any]]: ...
def pair_list_from_embeddings(
    pairs: Sequence[tuple[str, str]],
    embeddings: Sequence[Any],
    *,
    metadata: Sequence[dict[str, Any] | None] | None = None,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> list[dict[str, Any]]: ...
def all_vs_all_from_structure(
    inputs: Sequence[Any],
    include_self: bool = False,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def all_vs_all_from_coords(
    coords: Sequence[Any],
    include_self: bool = False,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def all_vs_all_to_tsv_from_embeddings(
    embeddings: Sequence[Any],
    output_path: str,
    metadata: Sequence[dict[str, Any] | None] | None = None,
    include_self: bool = False,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def all_vs_all_to_tsv_from_structure(
    inputs: Sequence[Any],
    output_path: str,
    include_self: bool = False,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def all_vs_all_to_tsv_from_coords(
    coords: Sequence[Any],
    output_path: str,
    include_self: bool = False,
    gap_open: float | None = None,
    gap_extension: float | None = None,
    package: str | None = None,
    thread_count: int = 0,
    mode: str = "hard",
    temperature: float = 1.0,
) -> dict[str, Any]: ...
def score_alignment_from_structure(
    query: Any,
    target: Any,
    *,
    correspondences: Sequence[Any],
    chain_id: str | None = None,
    chain_index: int | None = None,
    pdb_model_id: str | None = None,
    pdb_model_index: int | None = None,
) -> dict[str, Any]: ...
def _buffer_info(obj: Any) -> dict[str, Any]: ...
def _accepts_float32_2d(obj: Any) -> bool: ...
