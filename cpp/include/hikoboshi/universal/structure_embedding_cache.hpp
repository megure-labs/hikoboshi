#ifndef HIKOBOSHI_UNIVERSAL_STRUCTURE_EMBEDDING_CACHE_HPP
#define HIKOBOSHI_UNIVERSAL_STRUCTURE_EMBEDDING_CACHE_HPP

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::universal {

/// Optional encoding cache for structure/coordinate batch workflows.
/// Implementations must be thread-safe and bind model weights, encoder options,
/// code/build identity and numerical mode, as well as all consumed input values.
/// The owner outlives the synchronous engine call; views cannot be retained.
/// A hit fills every output element. A miss leaves output unspecified; the
/// engine overwrites it before calling store. Errors stop the workflow.
class StructureEmbeddingCache {
 public:
  virtual ~StructureEmbeddingCache() = default;
  [[nodiscard]] virtual Status load(const StructureView& structure,
                                    Span<float> output, bool& hit) = 0;
  [[nodiscard]] virtual Status store(const StructureView& structure,
                                     Span<const float> values) = 0;
};

}  // namespace hikoboshi::universal
#endif
