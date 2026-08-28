#include <hikoboshi/algorithms/metrics.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace hikoboshi::algorithms {
namespace {

Point3 add(Point3 a, Point3 b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Point3 subtract(Point3 a, Point3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Point3 scale(Point3 a, double value) noexcept {
  return {a.x * value, a.y * value, a.z * value};
}

double squared_norm(Point3 a) noexcept {
  return a.x * a.x + a.y * a.y + a.z * a.z;
}

void identity4(double matrix[4][4]) noexcept {
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) {
      matrix[r][c] = r == c ? 1.0 : 0.0;
    }
  }
}

void jacobi_largest_eigenvector(const double input[4][4],
                                double eigenvector[4]) noexcept {
  double a[4][4]{};
  double v[4][4]{};
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) {
      a[r][c] = input[r][c];
    }
  }
  identity4(v);

  for (std::size_t sweep = 0; sweep < 64; ++sweep) {
    std::size_t p = 0;
    std::size_t q = 1;
    double max_offdiag = std::fabs(a[p][q]);
    for (std::size_t r = 0; r < 4; ++r) {
      for (std::size_t c = r + 1; c < 4; ++c) {
        const double value = std::fabs(a[r][c]);
        if (value > max_offdiag) {
          max_offdiag = value;
          p = r;
          q = c;
        }
      }
    }
    if (max_offdiag < 1.0e-12) {
      break;
    }

    const double app = a[p][p];
    const double aqq = a[q][q];
    const double apq = a[p][q];
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);

    for (std::size_t r = 0; r < 4; ++r) {
      if (r == p || r == q) {
        continue;
      }
      const double arp = a[r][p];
      const double arq = a[r][q];
      a[r][p] = cosine * arp - sine * arq;
      a[p][r] = a[r][p];
      a[r][q] = sine * arp + cosine * arq;
      a[q][r] = a[r][q];
    }

    a[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq +
              sine * sine * aqq;
    a[q][q] = sine * sine * app + 2.0 * sine * cosine * apq +
              cosine * cosine * aqq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    for (std::size_t r = 0; r < 4; ++r) {
      const double vrp = v[r][p];
      const double vrq = v[r][q];
      v[r][p] = cosine * vrp - sine * vrq;
      v[r][q] = sine * vrp + cosine * vrq;
    }
  }

  std::size_t largest = 0;
  for (std::size_t i = 1; i < 4; ++i) {
    if (a[i][i] > a[largest][largest]) {
      largest = i;
    }
  }
  double norm = 0.0;
  for (std::size_t r = 0; r < 4; ++r) {
    eigenvector[r] = v[r][largest];
    norm += eigenvector[r] * eigenvector[r];
  }
  norm = std::sqrt(norm);
  if (norm == 0.0) {
    eigenvector[0] = 1.0;
    eigenvector[1] = 0.0;
    eigenvector[2] = 0.0;
    eigenvector[3] = 0.0;
    return;
  }
  for (std::size_t r = 0; r < 4; ++r) {
    eigenvector[r] /= norm;
  }
}

void quaternion_to_rotation(const double q[4], double rotation[9]) noexcept {
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];
  rotation[0] = 1.0 - 2.0 * (y * y + z * z);
  rotation[1] = 2.0 * (x * y - w * z);
  rotation[2] = 2.0 * (x * z + w * y);
  rotation[3] = 2.0 * (x * y + w * z);
  rotation[4] = 1.0 - 2.0 * (x * x + z * z);
  rotation[5] = 2.0 * (y * z - w * x);
  rotation[6] = 2.0 * (x * z - w * y);
  rotation[7] = 2.0 * (y * z + w * x);
  rotation[8] = 1.0 - 2.0 * (x * x + y * y);
}

bool observed_ca(const hikoboshi::universal::StructureView& structure,
                 std::size_t residue) noexcept {
  const std::size_t atom =
      static_cast<std::size_t>(hikoboshi::universal::CanonicalAtom::CA);
  return structure.atom_sources
             .data[residue * hikoboshi::universal::kCanonicalAtomCount + atom] ==
         hikoboshi::universal::AtomSource::Observed;
}

Point3 ca_point(const hikoboshi::universal::StructureView& structure,
                std::size_t residue) noexcept {
  const std::size_t atom =
      static_cast<std::size_t>(hikoboshi::universal::CanonicalAtom::CA);
  const std::size_t offset =
      (residue * hikoboshi::universal::kCanonicalAtomCount + atom) *
      hikoboshi::universal::kCoordinateAxisCount;
  return {structure.coordinates.data[offset],
          structure.coordinates.data[offset + 1],
          structure.coordinates.data[offset + 2]};
}

template <typename ForEachPair>
KabschResult kabsch_from_pairs(std::size_t pair_count,
                               ForEachPair for_each_pair) noexcept {
  KabschResult result{};
  result.pair_count = pair_count;
  if (pair_count < 3) {
    result.reason =
        hikoboshi::universal::MetricInvalidReason::InsufficientAlignedPairs;
    return result;
  }

  Point3 query_sum{};
  Point3 target_sum{};
  for_each_pair([&](Point3 query, Point3 target) noexcept {
    query_sum = add(query_sum, query);
    target_sum = add(target_sum, target);
  });
  const Point3 query_centroid =
      scale(query_sum, 1.0 / static_cast<double>(pair_count));
  const Point3 target_centroid =
      scale(target_sum, 1.0 / static_cast<double>(pair_count));

  double sxx = 0.0;
  double sxy = 0.0;
  double sxz = 0.0;
  double syx = 0.0;
  double syy = 0.0;
  double syz = 0.0;
  double szx = 0.0;
  double szy = 0.0;
  double szz = 0.0;

  for_each_pair([&](Point3 query, Point3 target) noexcept {
    const Point3 q = subtract(query, query_centroid);
    const Point3 t = subtract(target, target_centroid);
    sxx += t.x * q.x;
    sxy += t.x * q.y;
    sxz += t.x * q.z;
    syx += t.y * q.x;
    syy += t.y * q.y;
    syz += t.y * q.z;
    szx += t.z * q.x;
    szy += t.z * q.y;
    szz += t.z * q.z;
  });

  const double k[4][4] = {
      {sxx + syy + szz, syz - szy, szx - sxz, sxy - syx},
      {syz - szy, sxx - syy - szz, sxy + syx, szx + sxz},
      {szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy},
      {sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz},
  };
  double q[4]{};
  jacobi_largest_eigenvector(k, q);
  quaternion_to_rotation(q, result.transform.rotation);

  const Point3 rotated_target_centroid =
      apply_transform(result.transform, target_centroid);
  result.transform.translation =
      subtract(query_centroid, rotated_target_centroid);

  double sum_squared = 0.0;
  for_each_pair([&](Point3 query, Point3 target) noexcept {
    const Point3 transformed = apply_transform(result.transform, target);
    sum_squared += squared_norm(subtract(query, transformed));
  });

  result.valid = true;
  result.reason = hikoboshi::universal::MetricInvalidReason::None;
  result.rmsd = std::sqrt(sum_squared / static_cast<double>(pair_count));
  return result;
}

std::size_t observed_aligned_ca_pair_count(
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target) noexcept {
  std::size_t count = 0;
  AlignedCaPair pair{};
  for (const auto& step : path.steps) {
    if (load_aligned_observed_ca_pair(step, query, target, pair)) {
      ++count;
    }
  }
  return count;
}

}  // namespace

Point3 apply_transform(const KabschTransform& transform, Point3 point) noexcept {
  return {
      transform.rotation[0] * point.x + transform.rotation[1] * point.y +
          transform.rotation[2] * point.z + transform.translation.x,
      transform.rotation[3] * point.x + transform.rotation[4] * point.y +
          transform.rotation[5] * point.z + transform.translation.y,
      transform.rotation[6] * point.x + transform.rotation[7] * point.y +
          transform.rotation[8] * point.z + transform.translation.z,
  };
}

bool has_complete_structure_coordinates(
    const hikoboshi::universal::StructureView& structure) noexcept {
  const std::size_t atom_count =
      structure.residue_count * hikoboshi::universal::kCanonicalAtomCount;
  return structure.coordinates.data != nullptr &&
         structure.coordinates.size >=
             atom_count * hikoboshi::universal::kCoordinateAxisCount &&
         structure.atom_sources.data != nullptr &&
         structure.atom_sources.size >= atom_count;
}

bool extract_aligned_pair_index(
    const hikoboshi::universal::AlignmentStep& step,
    AlignedPairIndex& pair) noexcept {
  if (step.query_index < 0 || step.target_index < 0) {
    return false;
  }
  pair.query_index = static_cast<std::size_t>(step.query_index);
  pair.target_index = static_cast<std::size_t>(step.target_index);
  return true;
}

bool load_aligned_observed_ca_pair(
    AlignedPairIndex index,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target,
    AlignedCaPair& pair) noexcept {
  if (!has_complete_structure_coordinates(query) ||
      !has_complete_structure_coordinates(target)) {
    return false;
  }
  if (index.query_index >= query.residue_count ||
      index.target_index >= target.residue_count) {
    return false;
  }
  if (!observed_ca(query, index.query_index) ||
      !observed_ca(target, index.target_index)) {
    return false;
  }
  pair.query = ca_point(query, index.query_index);
  pair.target = ca_point(target, index.target_index);
  return true;
}

void extract_aligned_pairs(const AlignedPairExtractionRequest& request,
                           AlignedPairExtractionOutput& output) noexcept {
  output.aligned_pair_count = 0;
  output.observed_pair_count = 0;
  output.truncated = false;
  if (request.path == nullptr) {
    return;
  }

  const bool can_mark_observed =
      has_complete_structure_coordinates(request.query_structure) &&
      has_complete_structure_coordinates(request.target_structure);
  AlignedCaPair ca_pair{};
  for (const auto& step : request.path->steps) {
    AlignedPairIndex index{};
    if (!extract_aligned_pair_index(step, index)) {
      continue;
    }

    const std::size_t write_index = output.aligned_pair_count;
    const bool observed =
        can_mark_observed &&
        load_aligned_observed_ca_pair(index, request.query_structure,
                                      request.target_structure, ca_pair);
    if (observed) {
      ++output.observed_pair_count;
    }
    if (write_index < output.pair_indices.size &&
        output.pair_indices.data != nullptr) {
      output.pair_indices.data[write_index] = index;
    } else {
      output.truncated = true;
    }
    if (write_index < output.observed_pair_mask.size &&
        output.observed_pair_mask.data != nullptr) {
      output.observed_pair_mask.data[write_index] = observed ? 1U : 0U;
    } else if (output.observed_pair_mask.size != 0U ||
               output.observed_pair_mask.data != nullptr) {
      output.truncated = true;
    }
    ++output.aligned_pair_count;
  }
}

bool load_aligned_observed_ca_pair(
    const hikoboshi::universal::AlignmentStep& step,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target,
    AlignedCaPair& pair) noexcept {
  AlignedPairIndex index{};
  if (!extract_aligned_pair_index(step, index)) {
    return false;
  }
  return load_aligned_observed_ca_pair(index, query, target, pair);
}

KabschResult kabsch_superpose(
    hikoboshi::universal::Span<const Point3> query_points,
    hikoboshi::universal::Span<const Point3> target_points) noexcept {
  KabschResult result{};
  result.pair_count = std::min(query_points.size, target_points.size);
  if (query_points.data == nullptr || target_points.data == nullptr ||
      query_points.size != target_points.size) {
    result.reason =
        hikoboshi::universal::MetricInvalidReason::MissingStructureMetadata;
    return result;
  }
  return kabsch_from_pairs(query_points.size,
                           [&](auto&& receiver) noexcept {
    for (std::size_t i = 0; i < query_points.size; ++i) {
      receiver(query_points.data[i], target_points.data[i]);
    }
  });
}

KabschResult kabsch_superpose_aligned_ca(
    const hikoboshi::universal::AlignmentPath& path,
    const hikoboshi::universal::StructureView& query,
    const hikoboshi::universal::StructureView& target) noexcept {
  KabschResult result{};
  if (!has_complete_structure_coordinates(query) ||
      !has_complete_structure_coordinates(target)) {
    result.reason =
        hikoboshi::universal::MetricInvalidReason::MissingStructureMetadata;
    return result;
  }
  const std::size_t pair_count =
      observed_aligned_ca_pair_count(path, query, target);
  return kabsch_from_pairs(pair_count, [&](auto&& receiver) noexcept {
    AlignedCaPair pair{};
    for (const auto& step : path.steps) {
      if (load_aligned_observed_ca_pair(step, query, target, pair)) {
        receiver(pair.query, pair.target);
      }
    }
  });
}

}  // namespace hikoboshi::algorithms
