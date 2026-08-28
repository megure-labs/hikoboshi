#include <hikoboshi/algorithms/metrics.hpp>

namespace hikoboshi::algorithms {
using hikoboshi::universal::MetricValue;
using hikoboshi::universal::StructureView;

MetricValue compute_rmsd(const hikoboshi::universal::AlignmentPath& path,
                         const StructureView& query,
                         const StructureView& target) noexcept {
  const KabschResult kabsch = kabsch_superpose_aligned_ca(path, query, target);
  if (!kabsch.valid) {
    return invalid_metric(kabsch.reason);
  }
  return valid_metric(kabsch.rmsd);
}

}  // namespace hikoboshi::algorithms
