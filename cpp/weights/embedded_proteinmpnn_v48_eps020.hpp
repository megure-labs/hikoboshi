#ifndef HIKOBOSHI_WEIGHTS_EMBEDDED_PROTEINMPNN_V48_EPS020_HPP
#define HIKOBOSHI_WEIGHTS_EMBEDDED_PROTEINMPNN_V48_EPS020_HPP

#include <cstddef>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/weights.hpp>
#include <hikoboshi/weights/provider.hpp>

#include "generated/proteinmpnn_v48_eps020_blob.hpp"

namespace hikoboshi::weights::detail {

const hikoboshi::universal::WeightsHandle&
embedded_proteinmpnn_v48_eps020_handle() noexcept;

bool embedded_proteinmpnn_v48_eps020_manifest_matches() noexcept;

bool validate_proteinmpnn_v48_eps020_generated_tensors(
    hikoboshi::universal::Span<
        const hikoboshi::weights::generated::proteinmpnn_v48_eps020::TensorBlobInfo>
        tensors,
    PackageValidationBuffer& buffer,
    std::size_t& diagnostic_count) noexcept;

}  // namespace hikoboshi::weights::detail

#endif  // HIKOBOSHI_WEIGHTS_EMBEDDED_PROTEINMPNN_V48_EPS020_HPP
