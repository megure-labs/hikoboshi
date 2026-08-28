#ifndef HIKOBOSHI_DISPATCH_BACKEND_TAG_HPP
#define HIKOBOSHI_DISPATCH_BACKEND_TAG_HPP

#include <hikoboshi/universal/backend.hpp>

namespace hikoboshi::dispatch {

struct ScalarTag {};
struct Sse4Tag {};
struct Avx2Tag {};
struct Avx512Tag {};
struct NeonTag {};
struct SveTag {};
struct CudaTag {};
struct HipTag {};
struct MetalTag {};
struct VulkanTag {};
struct OpenClTag {};

// Per-op parity-mode dispatch tags. Orthogonal to backend tags.
//
// `StrictParityTag` selects the bit-identity-to-reference kernel for ops
// whose backend exposes a strict-mode entry (e.g. the PyTorch CUDA cuBLAS
// reduction-tree match for GEMM). `FastParityTag` selects the in-tolerance
// kernel that promises only the public tolerance contract (1e-4 max abs
// for GEMM) but may produce a different reduction tree.
struct StrictParityTag {};
struct FastParityTag {};

inline constexpr bool selects_scalar(hikoboshi::universal::Backend backend) noexcept {
  return backend == hikoboshi::universal::Backend::Auto ||
         backend == hikoboshi::universal::Backend::Scalar;
}

}  // namespace hikoboshi::dispatch

#endif  // HIKOBOSHI_DISPATCH_BACKEND_TAG_HPP
