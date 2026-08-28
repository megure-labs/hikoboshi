#ifndef HIKOBOSHI_WEIGHTS_EMBEDDED_MPNN_D64_HPP
#define HIKOBOSHI_WEIGHTS_EMBEDDED_MPNN_D64_HPP

#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/provider.hpp>

#include "generated/mpnn_d64_blob.hpp"

namespace hikoboshi::weights::detail {

const hikoboshi::universal::WeightsHandle& embedded_mpnn_d64_handle() noexcept;
bool embedded_mpnn_d64_manifest_matches() noexcept;
bool validate_mpnn_d64_generated_tensors(
    hikoboshi::universal::Span<
        const hikoboshi::weights::generated::mpnn_d64::TensorBlobInfo> tensors,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept;

}  // namespace hikoboshi::weights::detail

#endif  // HIKOBOSHI_WEIGHTS_EMBEDDED_MPNN_D64_HPP
