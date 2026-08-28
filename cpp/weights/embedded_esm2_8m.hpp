#ifndef HIKOBOSHI_WEIGHTS_EMBEDDED_ESM2_8M_HPP
#define HIKOBOSHI_WEIGHTS_EMBEDDED_ESM2_8M_HPP

// Internal surface for the hikoboshi-esm2-8m compiled package.
//
// The architecture descriptor + tokenizer table landed in
// `esm2-8m-architecture-design`; the safetensors payload, typed
// prepared-state binding, and runtime validation helpers landed in
// `esm2-8m-weights-package` (this packet). `embedded_esm2_8m_handle()`
// returns a `WeightsHandle` whose opaque pointer references the typed
// `Esm2Weights` view consumed by the modules-layer forward pass; the
// metadata view inside the handle carries the real source/payload
// SHA-256 records.

#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include "generated/esm2_8m_blob.hpp"

namespace hikoboshi::weights::detail {

/// Return the read-only weights handle for the embedded hikoboshi-esm2-8m
/// package. The metadata view inside the handle stays valid for the
/// process lifetime; tensor spans reference the embedded safetensors
/// data section.
const hikoboshi::universal::WeightsHandle& embedded_esm2_8m_handle() noexcept;

/// Return the ESM2-8M ASCII token table as a span. Slot order matches
/// Casey's compacted local alphabet (see
/// `hikoboshi_train/esm2_encoder.py`): 20 canonical amino acids at
/// indices 0-19, 5 non-standard residues at 20-24, then PAD, CLS/BOS,
/// EOS, MASK at 25-28. The four FAIR-only ESM2 tokens (`<unk>`, `.`,
/// `-`, `<null_1>`) are intentionally dropped.
hikoboshi::universal::Span<const std::string_view>
embedded_esm2_8m_tokenizer_table() noexcept;

/// Validate the embedded hikoboshi-esm2-8m runtime tensor table.
///
/// Walks the generated tensor metadata, asserts every expected
/// architecture slot is present with the right dtype/rank/shape/byte
/// length, and confirms each tensor's byte range stays inside the
/// safetensors data section with float-aligned offsets and contiguous
/// row-major strides. Emits stage-4
/// (TensorTableRolesShapesDtypes) diagnostics into `buffer` for any
/// mismatch.
bool validate_esm2_8m_generated_tensors(
    hikoboshi::universal::Span<
        const hikoboshi::weights::generated::esm2_8m::TensorBlobInfo> tensors,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept;

/// Confirm the embedded manifest matches the generated tensor table.
///
/// Returns true when the public `default_esm2_8m_manifest()` view
/// declares the same model identity, payload checksum, gap defaults,
/// and per-tensor records the generated blob carries. Used by the
/// provider validation pipeline to guard against an out-of-sync
/// manifest/blob pair.
bool embedded_esm2_8m_manifest_matches() noexcept;

}  // namespace hikoboshi::weights::detail

#endif  // HIKOBOSHI_WEIGHTS_EMBEDDED_ESM2_8M_HPP
