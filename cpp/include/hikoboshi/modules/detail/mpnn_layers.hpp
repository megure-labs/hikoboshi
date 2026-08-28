#ifndef HIKOBOSHI_MODULES_DETAIL_MPNN_LAYERS_HPP
#define HIKOBOSHI_MODULES_DETAIL_MPNN_LAYERS_HPP

#include <cstddef>

#include <hikoboshi/universal/detail/mpnn_d64_weights.hpp>

namespace hikoboshi::modules::detail {

inline constexpr std::size_t kMpnn64HiddenDimension =
    hikoboshi::universal::detail::kMpnn64HiddenDimension;
inline constexpr std::size_t kMpnn64AtomCount =
    hikoboshi::universal::detail::kMpnn64AtomCount;
inline constexpr std::size_t kMpnn64AxisCount =
    hikoboshi::universal::detail::kMpnn64AxisCount;
inline constexpr std::size_t kMpnn64AtomSourceFeatureCount =
    hikoboshi::universal::detail::kMpnn64AtomSourceFeatureCount;
inline constexpr std::size_t kMpnn64ResidueFeatureCount =
    hikoboshi::universal::detail::kMpnn64ResidueFeatureCount;
inline constexpr std::size_t kMpnn64EdgeFeatureCount =
    hikoboshi::universal::detail::kMpnn64EdgeFeatureCount;
inline constexpr std::size_t kMpnn64MessageInputDimension =
    hikoboshi::universal::detail::kMpnn64MessageInputDimension;
inline constexpr std::size_t kMpnn64FfnHiddenDimension =
    hikoboshi::universal::detail::kMpnn64FfnHiddenDimension;
inline constexpr std::size_t kMpnn64PositionalEncodingCount =
    hikoboshi::universal::detail::kMpnn64PositionalEncodingCount;
inline constexpr std::size_t kMpnn64PositionalEncodingInputDimension =
    hikoboshi::universal::detail::kMpnn64PositionalEncodingInputDimension;
inline constexpr float kMpnn64LayerNormEpsilon =
    hikoboshi::universal::detail::kMpnn64LayerNormEpsilon;

using Mpnn64InputWeights = hikoboshi::universal::detail::Mpnn64InputWeights;
using Mpnn64TensorWeights = hikoboshi::universal::detail::Mpnn64TensorWeights;
using Mpnn64LinearWeights = hikoboshi::universal::detail::Mpnn64LinearWeights;
using Mpnn64NormWeights = hikoboshi::universal::detail::Mpnn64NormWeights;
using Mpnn64EdgeEmbeddingWeights =
    hikoboshi::universal::detail::Mpnn64EdgeEmbeddingWeights;
using Mpnn64FeedForwardWeights =
    hikoboshi::universal::detail::Mpnn64FeedForwardWeights;
using Mpnn64LayerWeights = hikoboshi::universal::detail::Mpnn64LayerWeights;
using Mpnn64Weights = hikoboshi::universal::detail::Mpnn64Weights;

}  // namespace hikoboshi::modules::detail

#endif  // HIKOBOSHI_MODULES_DETAIL_MPNN_LAYERS_HPP
