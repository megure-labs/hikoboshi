"""Python adapter for the Hikoboshi 0.1.0 public API."""

from typing import Any

from .all_vs_all import all_vs_all
from .encode import encode
from .exceptions import (
    FailedPreconditionError,
    InternalError,
    InvalidArgumentError,
    HikoboshiError,
    UnavailableError,
    UnimplementedError,
)
from .info import info, version_info
from .inverse_fold import inverse_fold
from .pair_list import (
    pair_list_from_coords,
    pair_list_from_embeddings,
    pair_list_from_sequence,
    pair_list_from_structure,
)
from .pairwise import pairwise
from .results import (
    AlignmentPath,
    AlignmentStep,
    AllVsAllRecord,
    AllVsAllResult,
    EncodeResult,
    InverseFoldLogProbsArtifact,
    InverseFoldResult,
    InverseFoldSequenceResult,
    PairwiseMetrics,
    PairwiseResult,
)
from .score_alignment import ScoreAlignmentResult, score_alignment

__version__ = "0.1.0"

ALIGNMENT_GAP_SENTINEL = -1


def __getattr__(name: str) -> Any:
    if name == "Tensor":
        from ._arrays import load_core

        tensor = load_core().Tensor
        globals()["Tensor"] = tensor
        return tensor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "ALIGNMENT_GAP_SENTINEL",
    "Tensor",
    "encode",
    "pairwise",
    "inverse_fold",
    "pair_list_from_sequence",
    "pair_list_from_structure",
    "pair_list_from_coords",
    "pair_list_from_embeddings",
    "all_vs_all",
    "score_alignment",
    "info",
    "version_info",
    "HikoboshiError",
    "InvalidArgumentError",
    "FailedPreconditionError",
    "UnavailableError",
    "UnimplementedError",
    "InternalError",
    "AlignmentStep",
    "AlignmentPath",
    "PairwiseMetrics",
    "PairwiseResult",
    "AllVsAllRecord",
    "AllVsAllResult",
    "EncodeResult",
    "InverseFoldSequenceResult",
    "InverseFoldLogProbsArtifact",
    "InverseFoldResult",
    "ScoreAlignmentResult",
]
