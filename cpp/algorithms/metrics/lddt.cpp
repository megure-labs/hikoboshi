#include <hikoboshi/algorithms/metrics.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace hikoboshi::algorithms {
namespace {

using hikoboshi::universal::AtomSource;
using hikoboshi::universal::CanonicalAtom;
using hikoboshi::universal::MetricInvalidReason;
using hikoboshi::universal::MetricValue;
using hikoboshi::universal::StructureView;

constexpr double kThresholds[4] = {0.5, 1.0, 2.0, 4.0};
constexpr std::size_t kThresholdCount = 4;
constexpr double kThresholdEpsilon = 1.0e-12;
constexpr std::int32_t kUnaligned = -1;

double distance(Point3 a, Point3 b) noexcept {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool observed_ca(const StructureView& structure, std::size_t residue) noexcept {
  const std::size_t atom = static_cast<std::size_t>(CanonicalAtom::CA);
  const std::size_t offset =
      residue * hikoboshi::universal::kCanonicalAtomCount + atom;
  return structure.atom_sources.data[offset] == AtomSource::Observed;
}

Point3 ca_point(const StructureView& structure, std::size_t residue) noexcept {
  const std::size_t atom = static_cast<std::size_t>(CanonicalAtom::CA);
  const std::size_t offset =
      (residue * hikoboshi::universal::kCanonicalAtomCount + atom) *
      hikoboshi::universal::kCoordinateAxisCount;
  return {structure.coordinates.data[offset],
          structure.coordinates.data[offset + 1],
          structure.coordinates.data[offset + 2]};
}

std::size_t count_threshold_passes(double delta) noexcept {
  std::size_t passes = 0;
  for (double threshold : kThresholds) {
    if (delta < threshold || std::fabs(delta - threshold) < kThresholdEpsilon) {
      ++passes;
    }
  }
  return passes;
}

LddtMetrics make_invalid(MetricInvalidReason reason) noexcept {
  const MetricValue invalid = invalid_metric(reason);
  return {invalid, invalid, invalid, invalid, invalid, invalid};
}

// Canonical (Mariani) directional lDDT counters for one reference structure.
//
// Iterates over all CA-observed residue pairs (i,j) of the reference within
// `r0`, accumulating:
//   * `pair_count_in_R0`     -- denominator for the directional `lddt_byX`.
//   * `aligned_aligned_in_R0` -- aligned-aligned subset, i.e., pairs where both
//                                residues have aligned partners with observed
//                                CA in the model. Drives `coverage_byX` and
//                                acts as the denominator for the A-reference
//                                `lddt_aln`.
//   * `pass_count`            -- threshold-pass total over the aligned-aligned
//                                subset. Unaligned pairs contribute 0 passes
//                                so they only inflate the canonical denominator
//                                without ever crediting numerator passes.
struct DirectionalCounters {
  std::size_t pair_count_in_R0 = 0;
  std::size_t aligned_aligned_in_R0 = 0;
  std::size_t pass_count = 0;
};

// Contact reuse is bounded per calling thread and keyed by consumed contents,
// never by StructureView pointers. Callers may mutate or replace their buffers
// between calls. Very large/dense structures retain the direct traversal.
constexpr std::size_t kContactCacheEntries = 4;
constexpr std::size_t kContactCacheMaxResidues = 4096;
constexpr std::size_t kContactCacheMaxContacts = 32768;

struct ReferenceContact {
  std::uint32_t first;
  std::uint32_t second;
  double distance;
};

struct ReferenceContacts {
  // Three float bit patterns and an observed flag for each residue.
  std::vector<std::uint32_t> ca_snapshot;
  std::vector<ReferenceContact> contacts;
  double radius = 0.0;
  bool valid = false;
  bool prepared = false;
  bool reuse_seen = false;
  bool overflow = false;

  bool matches(const StructureView& structure, double r0) const noexcept {
    if (!valid || radius != r0 || ca_snapshot.size() != structure.residue_count * 4)
      return false;
    for (std::size_t i = 0; i < structure.residue_count; ++i) {
      const bool observed = observed_ca(structure, i);
      if (ca_snapshot[i * 4 + 3] != static_cast<std::uint32_t>(observed))
        return false;
      const std::size_t offset =
          (i * hikoboshi::universal::kCanonicalAtomCount +
           static_cast<std::size_t>(CanonicalAtom::CA)) * 3;
      if (observed && std::memcmp(ca_snapshot.data() + i * 4,
                                 structure.coordinates.data + offset,
                                 3 * sizeof(float)) != 0)
        return false;
    }
    return true;
  }
};

ReferenceContacts* reference_contacts(const StructureView& structure,
                                      double r0) noexcept {
  if (structure.residue_count > kContactCacheMaxResidues || !std::isfinite(r0))
    return nullptr;
  struct Cache {
    ReferenceContacts entries[kContactCacheEntries];
    std::size_t next = 0;
  };
  thread_local Cache cache;
  for (auto& entry : cache.entries) {
    if (entry.matches(structure, r0)) {
      entry.reuse_seen = true;
      return &entry;
    }
  }
  auto& entry = cache.entries[cache.next];
  cache.next = (cache.next + 1) % kContactCacheEntries;
  entry.valid = false;
  entry.prepared = false;
  entry.reuse_seen = false;
  entry.overflow = false;
  entry.contacts.clear();
  const std::size_t n = structure.residue_count;
  try {
    entry.ca_snapshot.reserve(n * 4);
    entry.ca_snapshot.resize(n * 4);
    entry.contacts.reserve(std::min(kContactCacheMaxContacts,
                                    n == 0 ? 0 : n * (n - 1) / 2));
  } catch (const std::bad_alloc&) {
    // Reuse is optional; failure to allocate it must not invalidate a metric.
    return nullptr;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const bool observed = observed_ca(structure, i);
    entry.ca_snapshot[i * 4 + 3] = static_cast<std::uint32_t>(observed);
    const std::size_t offset =
        (i * hikoboshi::universal::kCanonicalAtomCount +
         static_cast<std::size_t>(CanonicalAtom::CA)) * 3;
    if (observed)
      std::memcpy(entry.ca_snapshot.data() + i * 4,
                  structure.coordinates.data + offset, 3 * sizeof(float));
  }
  entry.radius = r0;
  entry.valid = true;
  return &entry;
}

void accumulate_contact(DirectionalCounters& counters,
                        std::size_t i, std::size_t j, double d_ref,
                        const StructureView& model,
                        const std::int32_t* reference_to_model) noexcept {
  ++counters.pair_count_in_R0;
  const std::int32_t partner_i = reference_to_model[i];
  const std::int32_t partner_j = reference_to_model[j];
  if (partner_i == kUnaligned || partner_j == kUnaligned) return;
  const std::size_t mi = static_cast<std::size_t>(partner_i);
  const std::size_t mj = static_cast<std::size_t>(partner_j);
  if (mi >= model.residue_count || mj >= model.residue_count ||
      !observed_ca(model, mi) || !observed_ca(model, mj)) return;
  ++counters.aligned_aligned_in_R0;
  const double d_model = distance(ca_point(model, mi), ca_point(model, mj));
  counters.pass_count += count_threshold_passes(std::fabs(d_ref - d_model));
}

DirectionalCounters accumulate_direction(
    const StructureView& reference,
    const StructureView& model,
    const std::int32_t* reference_to_model,
    double r0) noexcept {
  DirectionalCounters counters{};
  ReferenceContacts* cached = reference_contacts(reference, r0);
  if (cached != nullptr && cached->prepared) {
    for (const auto& contact : cached->contacts)
      accumulate_contact(counters, contact.first, contact.second,
                         contact.distance, model, reference_to_model);
    return counters;
  }
  for (std::size_t i = 0; i < reference.residue_count; ++i) {
    if (!observed_ca(reference, i)) continue;
    const Point3 p_i = ca_point(reference, i);
    for (std::size_t j = i + 1; j < reference.residue_count; ++j) {
      if (!observed_ca(reference, j)) continue;
      const double d_ref = distance(p_i, ca_point(reference, j));
      if (d_ref > r0) continue;
      if (cached != nullptr && cached->reuse_seen && !cached->overflow) {
        if (cached->contacts.size() == kContactCacheMaxContacts) {
          cached->contacts.clear();
          cached->overflow = true;
        } else {
          // Capacity was reserved before traversal; this cannot allocate.
          cached->contacts.push_back({static_cast<std::uint32_t>(i),
                                      static_cast<std::uint32_t>(j), d_ref});
        }
      }
      accumulate_contact(counters, i, j, d_ref, model, reference_to_model);
    }
  }
  if (cached != nullptr)
    cached->prepared = cached->reuse_seen && !cached->overflow;
  return counters;
}

// Steady-state alignment-map scratch.
//
// Pairwise hot paths repeatedly compute_lddt on residue counts that are
// approximately constant (the same query/target structures, or all-vs-all pairs
// with similar protein lengths). Per-thread scratch grows to fit the largest
// pair seen and then re-uses its capacity, so steady-state calls never allocate
// from the global allocator and the API performance-smoke no-allocation gate
// stays clean.
struct AlignmentMapScratch {
  std::vector<std::int32_t> query_to_target;
  std::vector<std::int32_t> target_to_query;
};

AlignmentMapScratch& scratch() {
  thread_local AlignmentMapScratch storage;
  return storage;
}

bool ensure_capacity(std::vector<std::int32_t>& buffer,
                     std::size_t count) noexcept {
  if (buffer.size() >= count) {
    return true;
  }
  try {
    buffer.resize(count);
  } catch (const std::bad_alloc&) {
    return false;
  }
  return true;
}

}  // namespace

// Canonical (Mariani 2013 / OpenStructure) lDDT plus aligned-only and
// per-direction coverage decomposition.
//
// Returns the six-field schema:
//   * `lddt_byA`, `lddt_byB`: directional canonical lDDTs whose denominator is
//     the count of all reference-structure CA pair distances in `r0`.
//   * `lddt`: symmetric mean `(lddt_byA + lddt_byB) / 2`.
//   * `lddt_aln`: aligned-only lDDT computed once with A as reference; its
//     denominator is the aligned-aligned subset of pairs in `r0` of A.
//   * `coverage_byA`, `coverage_byB`: aligned-aligned-pairs / total-pairs in
//     each reference's `r0` graph; together with `lddt_aln` they recover
//     `lddt_byX` exactly for byA (and approximately for byB; near-exact for
//     similar structures).
LddtMetrics compute_lddt(const hikoboshi::universal::AlignmentPath& path,
                         const StructureView& query,
                         const StructureView& target,
                         double r0) noexcept {
  if (!has_complete_structure_coordinates(query) ||
      !has_complete_structure_coordinates(target)) {
    return make_invalid(MetricInvalidReason::MissingStructureMetadata);
  }

  AlignmentMapScratch& maps = scratch();
  if (!ensure_capacity(maps.query_to_target, query.residue_count) ||
      !ensure_capacity(maps.target_to_query, target.residue_count)) {
    return make_invalid(MetricInvalidReason::Unavailable);
  }
  std::fill(maps.query_to_target.begin(),
            maps.query_to_target.begin() +
                static_cast<std::ptrdiff_t>(query.residue_count),
            kUnaligned);
  std::fill(maps.target_to_query.begin(),
            maps.target_to_query.begin() +
                static_cast<std::ptrdiff_t>(target.residue_count),
            kUnaligned);

  for (const auto& step : path.steps) {
    if (step.query_index < 0 || step.target_index < 0) {
      continue;
    }
    const std::size_t qi = static_cast<std::size_t>(step.query_index);
    const std::size_t ti = static_cast<std::size_t>(step.target_index);
    if (qi >= query.residue_count || ti >= target.residue_count) {
      continue;
    }
    maps.query_to_target[qi] = step.target_index;
    maps.target_to_query[ti] = step.query_index;
  }

  const DirectionalCounters by_a =
      accumulate_direction(query, target, maps.query_to_target.data(), r0);
  const DirectionalCounters by_b =
      accumulate_direction(target, query, maps.target_to_query.data(), r0);

  if (by_a.pair_count_in_R0 == 0 && by_b.pair_count_in_R0 == 0) {
    return make_invalid(MetricInvalidReason::InsufficientAlignedPairs);
  }

  LddtMetrics out{};
  const double denom_scale = static_cast<double>(kThresholdCount);

  if (by_a.pair_count_in_R0 != 0) {
    const double denom = static_cast<double>(by_a.pair_count_in_R0) * denom_scale;
    out.lddt_byA = valid_metric(static_cast<double>(by_a.pass_count) / denom);
    out.coverage_byA =
        valid_metric(static_cast<double>(by_a.aligned_aligned_in_R0) /
                     static_cast<double>(by_a.pair_count_in_R0));
  } else {
    out.lddt_byA = invalid_metric(MetricInvalidReason::InsufficientAlignedPairs);
    out.coverage_byA =
        invalid_metric(MetricInvalidReason::InsufficientAlignedPairs);
  }

  if (by_b.pair_count_in_R0 != 0) {
    const double denom = static_cast<double>(by_b.pair_count_in_R0) * denom_scale;
    out.lddt_byB = valid_metric(static_cast<double>(by_b.pass_count) / denom);
    out.coverage_byB =
        valid_metric(static_cast<double>(by_b.aligned_aligned_in_R0) /
                     static_cast<double>(by_b.pair_count_in_R0));
  } else {
    out.lddt_byB = invalid_metric(MetricInvalidReason::InsufficientAlignedPairs);
    out.coverage_byB =
        invalid_metric(MetricInvalidReason::InsufficientAlignedPairs);
  }

  if (out.lddt_byA.valid && out.lddt_byB.valid) {
    out.lddt = valid_metric((out.lddt_byA.value + out.lddt_byB.value) / 2.0);
  } else {
    out.lddt = invalid_metric(MetricInvalidReason::InsufficientAlignedPairs);
  }

  if (by_a.aligned_aligned_in_R0 != 0) {
    const double denom =
        static_cast<double>(by_a.aligned_aligned_in_R0) * denom_scale;
    out.lddt_aln =
        valid_metric(static_cast<double>(by_a.pass_count) / denom);
  } else {
    out.lddt_aln = invalid_metric(MetricInvalidReason::InsufficientAlignedPairs);
  }

  return out;
}

}  // namespace hikoboshi::algorithms
