#ifndef HIKOBOSHI_IO_PDB_WRITER_HPP
#define HIKOBOSHI_IO_PDB_WRITER_HPP

#include <cstddef>
#include <string>
#include <string_view>

#include <hikoboshi/api/requests.hpp>
#include <hikoboshi/api/results.hpp>
#include <hikoboshi/universal/backend.hpp>
#include <hikoboshi/universal/metrics.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

struct PdbTransform {
  double rotation[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double translation[3] = {0.0, 0.0, 0.0};
};

struct PdbWriterOptions {
  api::AlignmentOptions alignment{api::kDefaultGapOpen,
                                  api::kDefaultGapExtension};
  universal::Backend backend = universal::Backend::Auto;
  bool superpose_target = true;
  bool include_inferred_atoms = true;
  bool include_virtual_atoms = true;
  int metric_digits = 6;
};

struct PdbRenderResult {
  std::string contents;
  PdbTransform target_to_query{};
  bool transform_valid = false;
  universal::MetricInvalidReason transform_invalid_reason =
      universal::MetricInvalidReason::Unavailable;
  std::size_t transform_pair_count = 0;
};

[[nodiscard]] universal::Result<PdbRenderResult> render_superposed_pdb(
    const api::PairwiseResult& result,
    const universal::StructureView& query,
    const universal::StructureView& target,
    const PdbWriterOptions& options = {});

[[nodiscard]] universal::Status write_superposed_pdb(
    std::string_view path,
    const api::PairwiseResult& result,
    const universal::StructureView& query,
    const universal::StructureView& target,
    const PdbWriterOptions& options = {});

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_PDB_WRITER_HPP
