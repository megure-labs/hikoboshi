#ifndef HIKOBOSHI_UNIVERSAL_WEIGHTS_HPP
#define HIKOBOSHI_UNIVERSAL_WEIGHTS_HPP

/// @file
/// Public weight metadata and compatibility tensor views.

#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::universal {

/// User-facing metadata for a prepared model/scoring package tensor view.
struct ModelMetadataView {
  std::string_view model_name;
  std::string_view model_family;
  std::string_view model_version;
  std::size_t hidden_dimension;
  std::string_view source_identifier;
  std::string_view checksum;
};

/// Borrowed tensor collection for compatibility with weight-view consumers.
struct WeightsView {
  ModelMetadataView metadata;
  Span<const TensorView> tensors;
};

/// Compatibility view over package tensor storage.
///
/// Package-aware code should use `PackageHandle` as the top-level
/// model/scoring selection unit. This handle remains for callers that need
/// prepared tensor views only.
struct WeightsHandle {
  const void* opaque;
  const WeightsView* view;
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_WEIGHTS_HPP
