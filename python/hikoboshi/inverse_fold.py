from __future__ import annotations

from typing import Any

from ._arrays import load_core
from ._chain_resolve import (
    chain_selection,
    reject_unknown_kwargs,
    resolve_deprecated_model_kwargs,
    warn_deprecated_package_alias,
)
from .exceptions import raise_from_core
from .results import InverseFoldResult


class InverseFoldNamespace:
    def __call__(self, structure: Any, **kwargs: Any) -> InverseFoldResult:
        return self.from_structure(structure, **kwargs)

    def from_structure(
        self,
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
        **kwargs: Any,
    ) -> InverseFoldResult:
        pdb_model_id, pdb_model_index = resolve_deprecated_model_kwargs(
            kwargs,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
            context="inverse_fold.from_structure",
        )
        reject_unknown_kwargs(kwargs, context="inverse_fold.from_structure")
        warn_deprecated_package_alias(package)
        core = load_core()
        selection = chain_selection(
            chain_id=chain_id,
            chain_index=chain_index,
            pdb_model_id=pdb_model_id,
            pdb_model_index=pdb_model_index,
        )
        try:
            payload = core.inverse_fold_from_structure(
                structure,
                num_seqs=int(num_seqs),
                sampling_temp=float(sampling_temp),
                seed=int(seed),
                decode_order=decode_order,
                package=package,
                backbone_noise=float(backbone_noise),
                out_logprobs=out_logprobs,
                **selection,
            )
        except Exception as exc:  # noqa: BLE001 - maps extension exceptions.
            raise_from_core(exc)
        return InverseFoldResult.from_core(payload)

    def from_pdb(
        self,
        structure: Any,
        **kwargs: Any,
    ) -> InverseFoldResult:
        return self.from_structure(structure, **kwargs)

    def from_cif(
        self,
        structure: Any,
        **kwargs: Any,
    ) -> InverseFoldResult:
        return self.from_structure(structure, **kwargs)


inverse_fold = InverseFoldNamespace()

__all__ = ["InverseFoldNamespace", "inverse_fold"]
