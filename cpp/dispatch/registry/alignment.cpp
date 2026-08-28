#include <hikoboshi/dispatch/registry/alignment.hpp>

#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>

namespace hikoboshi::dispatch::registry {
namespace {

namespace hiko_u = hikoboshi::universal;

constexpr std::string_view kHardLocalAffineSwV1Id{"hard_local_affine_sw_v1"};

constexpr std::string_view kHardLocalAffineSwV1PrimitiveOps[] = {
    "smith_waterman_scalar",
    "traceback_scalar",
};

constexpr hiko_u::GapModel kHardLocalAffineSwV1GapFamilies[] = {
    hiko_u::GapModel::Affine,
};

}  // namespace

universal::Span<const RegisteredAlignmentRecord> alignment_registry() noexcept {
  static const RegisteredAlignmentRecord kRecords[] = {
      {
          hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1,
          kHardLocalAffineSwV1Id,
          {kHardLocalAffineSwV1PrimitiveOps,
           sizeof(kHardLocalAffineSwV1PrimitiveOps) /
               sizeof(kHardLocalAffineSwV1PrimitiveOps[0])},
          {kHardLocalAffineSwV1GapFamilies,
           sizeof(kHardLocalAffineSwV1GapFamilies) /
               sizeof(kHardLocalAffineSwV1GapFamilies[0])},
      },
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

const RegisteredAlignmentRecord* find_alignment(
    const hiko_u::AlignmentAlgorithmId kind) noexcept {
  const universal::Span<const RegisteredAlignmentRecord> records =
      alignment_registry();
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].kind == kind) {
      return &records.data[index];
    }
  }
  return nullptr;
}

}  // namespace hikoboshi::dispatch::registry
