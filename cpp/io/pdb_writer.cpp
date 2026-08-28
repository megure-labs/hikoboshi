#include <hikoboshi/io/pdb_writer.hpp>

#include <hikoboshi/io/all_vs_all_layout.hpp>
#include <hikoboshi/universal/version.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hikoboshi::io {
namespace {

struct Point3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct TransformFit {
  bool valid = false;
  universal::MetricInvalidReason reason =
      universal::MetricInvalidReason::Unavailable;
  PdbTransform transform{};
  double rmsd = 0.0;
  std::size_t pair_count = 0;
};

struct InputSelection {
  std::string input_id;
  std::string chain_id;
  std::string model_id;
};

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

Point3 centroid(const std::vector<Point3>& points) noexcept {
  Point3 sum{};
  for (const Point3 point : points) {
    sum = add(sum, point);
  }
  return scale(sum, 1.0 / static_cast<double>(points.size()));
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

Point3 apply_transform(const PdbTransform& transform, Point3 point) noexcept {
  return {
      transform.rotation[0] * point.x + transform.rotation[1] * point.y +
          transform.rotation[2] * point.z + transform.translation[0],
      transform.rotation[3] * point.x + transform.rotation[4] * point.y +
          transform.rotation[5] * point.z + transform.translation[1],
      transform.rotation[6] * point.x + transform.rotation[7] * point.y +
          transform.rotation[8] * point.z + transform.translation[2],
  };
}

TransformFit kabsch_superpose(const std::vector<Point3>& query_points,
                              const std::vector<Point3>& target_points) {
  TransformFit result{};
  result.pair_count = std::min(query_points.size(), target_points.size());
  if (query_points.size() != target_points.size()) {
    result.reason = universal::MetricInvalidReason::MissingStructureMetadata;
    return result;
  }
  if (query_points.size() < 3) {
    result.reason = universal::MetricInvalidReason::InsufficientAlignedPairs;
    return result;
  }

  const Point3 query_centroid = centroid(query_points);
  const Point3 target_centroid = centroid(target_points);

  double sxx = 0.0;
  double sxy = 0.0;
  double sxz = 0.0;
  double syx = 0.0;
  double syy = 0.0;
  double syz = 0.0;
  double szx = 0.0;
  double szy = 0.0;
  double szz = 0.0;

  for (std::size_t i = 0; i < query_points.size(); ++i) {
    const Point3 q = subtract(query_points[i], query_centroid);
    const Point3 t = subtract(target_points[i], target_centroid);
    sxx += t.x * q.x;
    sxy += t.x * q.y;
    sxz += t.x * q.z;
    syx += t.y * q.x;
    syy += t.y * q.y;
    syz += t.y * q.z;
    szx += t.z * q.x;
    szy += t.z * q.y;
    szz += t.z * q.z;
  }

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
  const Point3 translation = subtract(query_centroid, rotated_target_centroid);
  result.transform.translation[0] = translation.x;
  result.transform.translation[1] = translation.y;
  result.transform.translation[2] = translation.z;

  double sum_squared = 0.0;
  for (std::size_t i = 0; i < query_points.size(); ++i) {
    const Point3 transformed = apply_transform(result.transform, target_points[i]);
    sum_squared += squared_norm(subtract(query_points[i], transformed));
  }

  result.valid = true;
  result.reason = universal::MetricInvalidReason::None;
  result.rmsd = std::sqrt(sum_squared / static_cast<double>(query_points.size()));
  return result;
}

std::size_t coordinate_count(const universal::StructureView& structure) noexcept {
  return structure.residue_count * universal::kCanonicalAtomCount *
         universal::kCoordinateAxisCount;
}

std::size_t atom_source_count(const universal::StructureView& structure) noexcept {
  return structure.residue_count * universal::kCanonicalAtomCount;
}

bool has_structure_coordinates(
    const universal::StructureView& structure) noexcept {
  return structure.coordinates.data != nullptr &&
         structure.coordinates.size >= coordinate_count(structure) &&
         structure.atom_sources.data != nullptr &&
         structure.atom_sources.size >= atom_source_count(structure);
}

std::size_t atom_offset(std::size_t residue, std::size_t atom) noexcept {
  return residue * universal::kCanonicalAtomCount + atom;
}

bool observed_ca(const universal::StructureView& structure,
                 std::size_t residue) noexcept {
  const std::size_t atom = static_cast<std::size_t>(universal::CanonicalAtom::CA);
  return structure.atom_sources.data[atom_offset(residue, atom)] ==
         universal::AtomSource::Observed;
}

Point3 atom_point(const universal::StructureView& structure,
                  std::size_t residue,
                  std::size_t atom) noexcept {
  const std::size_t offset =
      atom_offset(residue, atom) * universal::kCoordinateAxisCount;
  return {structure.coordinates.data[offset],
          structure.coordinates.data[offset + 1],
          structure.coordinates.data[offset + 2]};
}

universal::Status collect_transform_points(
    const api::AlignmentPath& path,
    const universal::StructureView& query,
    const universal::StructureView& target,
    std::vector<Point3>& query_points,
    std::vector<Point3>& target_points) {
  if (!has_structure_coordinates(query) || !has_structure_coordinates(target)) {
    return universal::unavailable_status("superposed PDB requires structure coordinates");
  }
  query_points.clear();
  target_points.clear();
  for (const auto& step : path.steps) {
    if (step.query_index < 0 && step.target_index < 0) {
      return universal::invalid_argument_status("alignment path step cannot contain two gaps");
    }
    if (step.query_index < 0 || step.target_index < 0) {
      continue;
    }
    const std::size_t qi = static_cast<std::size_t>(step.query_index);
    const std::size_t ti = static_cast<std::size_t>(step.target_index);
    if (qi >= query.residue_count || ti >= target.residue_count) {
      return universal::invalid_argument_status("alignment path residue index is out of range");
    }
    if (!observed_ca(query, qi) || !observed_ca(target, ti)) {
      continue;
    }
    query_points.push_back(atom_point(
        query, qi, static_cast<std::size_t>(universal::CanonicalAtom::CA)));
    target_points.push_back(atom_point(
        target, ti, static_cast<std::size_t>(universal::CanonicalAtom::CA)));
  }
  return universal::ok_status();
}

std::string metric_invalid_reason(
    universal::MetricInvalidReason reason) {
  switch (reason) {
    case universal::MetricInvalidReason::None:
      return "none";
    case universal::MetricInvalidReason::Unavailable:
      return "unavailable";
    case universal::MetricInvalidReason::MissingSequenceMetadata:
      return "missing_sequence_metadata";
    case universal::MetricInvalidReason::MissingStructureMetadata:
      return "missing_structure_metadata";
    case universal::MetricInvalidReason::InsufficientAlignedPairs:
      return "insufficient_aligned_pairs";
    case universal::MetricInvalidReason::ZeroDenominator:
      return "zero_denominator";
    case universal::MetricInvalidReason::Unimplemented:
      return "unimplemented";
  }
  return "unknown_metric_invalid_reason";
}

std::string format_double(double value, int digits) {
  if (digits < 1) {
    digits = 1;
  }
  if (digits > 17) {
    digits = 17;
  }
  std::ostringstream out;
  out << std::setprecision(digits) << value;
  return out.str();
}

std::string format_metric(universal::MetricValue metric, int digits) {
  if (!metric.valid) {
    return "NA reason=" + metric_invalid_reason(metric.reason);
  }
  return format_double(metric.value, digits);
}

std::string backend_name(universal::Backend backend) {
  switch (backend) {
    case universal::Backend::Auto:
      return "auto";
    case universal::Backend::Scalar:
      return "scalar";
  }
  return "unknown";
}

std::string sanitized_or(std::string_view value, std::string_view fallback) {
  std::string sanitized = sanitize_identifier(value.empty() ? fallback : value);
  if (sanitized.empty()) {
    sanitized = std::string{fallback};
  }
  return sanitized;
}

InputSelection selection_from_structure(
    const universal::StructureView& structure,
    std::string_view fallback_id) {
  InputSelection selection{};
  selection.input_id = sanitized_or(structure.input_id, fallback_id);
  selection.chain_id = "A";
  selection.model_id = "1";
  if (structure.residues.data != nullptr && structure.residues.size > 0) {
    const universal::ResidueMetadataView& residue = structure.residues.data[0];
    selection.chain_id = sanitized_or(residue.chain_id, "A");
    if (!residue.model_id.empty()) {
      selection.model_id = sanitized_or(residue.model_id, "1");
    } else if (residue.model_index > 0) {
      selection.model_id = std::to_string(residue.model_index);
    }
  }
  return selection;
}

std::string residue_name_for_code(char code) {
  switch (code) {
    case 'A': return "ALA";
    case 'C': return "CYS";
    case 'D': return "ASP";
    case 'E': return "GLU";
    case 'F': return "PHE";
    case 'G': return "GLY";
    case 'H': return "HIS";
    case 'I': return "ILE";
    case 'K': return "LYS";
    case 'L': return "LEU";
    case 'M': return "MET";
    case 'N': return "ASN";
    case 'P': return "PRO";
    case 'Q': return "GLN";
    case 'R': return "ARG";
    case 'S': return "SER";
    case 'T': return "THR";
    case 'V': return "VAL";
    case 'W': return "TRP";
    case 'Y': return "TYR";
    default: return "UNK";
  }
}

std::string residue_name(const universal::StructureView& structure,
                         std::size_t residue_index) {
  if (structure.residues.data != nullptr &&
      residue_index < structure.residues.size &&
      !structure.residues.data[residue_index].original_residue_name.empty()) {
    std::string name{
        structure.residues.data[residue_index].original_residue_name.substr(0, 3)};
    while (name.size() < 3) {
      name.push_back(' ');
    }
    return name;
  }
  if (structure.residue_codes.data != nullptr &&
      residue_index < structure.residue_codes.size) {
    return residue_name_for_code(structure.residue_codes.data[residue_index]);
  }
  return "UNK";
}

char chain_id(const universal::StructureView& structure,
              std::size_t residue_index) {
  if (structure.residues.data != nullptr &&
      residue_index < structure.residues.size &&
      !structure.residues.data[residue_index].chain_id.empty()) {
    const char chain = structure.residues.data[residue_index].chain_id.front();
    return chain == '\0' || chain == ' ' ? 'A' : chain;
  }
  return 'A';
}

std::int32_t residue_number(const universal::StructureView& structure,
                            std::size_t residue_index) {
  if (structure.residues.data != nullptr &&
      residue_index < structure.residues.size &&
      structure.residues.data[residue_index].residue_number != 0) {
    return structure.residues.data[residue_index].residue_number;
  }
  return static_cast<std::int32_t>(residue_index + 1);
}

char insertion_code(const universal::StructureView& structure,
                    std::size_t residue_index) {
  if (structure.residues.data != nullptr &&
      residue_index < structure.residues.size) {
    const char code = structure.residues.data[residue_index].insertion_code;
    return code == '\0' ? ' ' : code;
  }
  return ' ';
}

const char* atom_name(std::size_t atom) noexcept {
  switch (static_cast<universal::CanonicalAtom>(atom)) {
    case universal::CanonicalAtom::N:
      return "N";
    case universal::CanonicalAtom::CA:
      return "CA";
    case universal::CanonicalAtom::C:
      return "C";
    case universal::CanonicalAtom::O:
      return "O";
    case universal::CanonicalAtom::CB:
      return "CB";
  }
  return "X";
}

char atom_element(std::size_t atom) noexcept {
  switch (static_cast<universal::CanonicalAtom>(atom)) {
    case universal::CanonicalAtom::N:
      return 'N';
    case universal::CanonicalAtom::O:
      return 'O';
    case universal::CanonicalAtom::CA:
    case universal::CanonicalAtom::C:
    case universal::CanonicalAtom::CB:
      return 'C';
  }
  return ' ';
}

bool atom_source_allowed(universal::AtomSource source,
                         const PdbWriterOptions& options) noexcept {
  switch (source) {
    case universal::AtomSource::Observed:
      return true;
    case universal::AtomSource::Inferred:
      return options.include_inferred_atoms;
    case universal::AtomSource::Virtual:
      return options.include_virtual_atoms;
    case universal::AtomSource::Missing:
      return false;
  }
  return false;
}

void append_atom_line(std::ostringstream& out,
                      int serial,
                      const char* name,
                      char element,
                      std::string_view resname,
                      char chain,
                      std::int32_t resseq,
                      char insertion,
                      Point3 point) {
  out << "ATOM  " << std::setw(5) << serial << ' ' << std::left
      << std::setw(4) << name << std::right << ' ' << std::setw(3)
      << resname << ' ' << chain << std::setw(4) << resseq << insertion
      << "   " << std::fixed << std::setprecision(3) << std::setw(8)
      << point.x << std::setw(8) << point.y << std::setw(8) << point.z
      << std::setw(6) << std::setprecision(2) << 1.00 << std::setw(6)
      << 0.00 << "          " << element << '\n';
}

void append_model(std::ostringstream& out,
                  int model_number,
                  const universal::StructureView& structure,
                  const PdbTransform& transform,
                  bool apply_superposition,
                  const PdbWriterOptions& options) {
  out << "MODEL     " << std::setw(4) << model_number << '\n';
  int serial = 1;
  for (std::size_t residue = 0; residue < structure.residue_count; ++residue) {
    for (std::size_t atom = 0; atom < universal::kCanonicalAtomCount; ++atom) {
      const universal::AtomSource source =
          structure.atom_sources.data[atom_offset(residue, atom)];
      if (!atom_source_allowed(source, options)) {
        continue;
      }
      Point3 point = atom_point(structure, residue, atom);
      if (apply_superposition) {
        point = apply_transform(transform, point);
      }
      append_atom_line(out, serial, atom_name(atom), atom_element(atom),
                       residue_name(structure, residue),
                       chain_id(structure, residue),
                       residue_number(structure, residue),
                       insertion_code(structure, residue), point);
      ++serial;
    }
  }
  out << "ENDMDL\n";
}

void append_transform_remark(std::ostringstream& out,
                             const PdbTransform& transform) {
  out << std::fixed << std::setprecision(6);
  out << "REMARK HIKOBOSHI_TRANSFORM_ROTATION "
      << transform.rotation[0] << ' ' << transform.rotation[1] << ' '
      << transform.rotation[2] << '\n';
  out << "REMARK HIKOBOSHI_TRANSFORM_ROTATION "
      << transform.rotation[3] << ' ' << transform.rotation[4] << ' '
      << transform.rotation[5] << '\n';
  out << "REMARK HIKOBOSHI_TRANSFORM_ROTATION "
      << transform.rotation[6] << ' ' << transform.rotation[7] << ' '
      << transform.rotation[8] << '\n';
  out << "REMARK HIKOBOSHI_TRANSFORM_TRANSLATION "
      << transform.translation[0] << ' ' << transform.translation[1] << ' '
      << transform.translation[2] << '\n';
}

void append_metric_remark(std::ostringstream& out,
                          std::string_view name,
                          universal::MetricValue metric,
                          int digits) {
  out << "REMARK HIKOBOSHI_METRIC " << name << ' '
      << format_metric(metric, digits) << '\n';
}

void append_remarks(std::ostringstream& out,
                    const api::PairwiseResult& result,
                    const InputSelection& query_selection,
                    const InputSelection& target_selection,
                    const PdbWriterOptions& options,
                    const TransformFit& fit) {
  out << "REMARK HIKOBOSHI_VERSION " << universal::kVersionMajor << '.'
      << universal::kVersionMinor << '.' << universal::kVersionPatch << '\n';
  out << "REMARK HIKOBOSHI_INPUT_QUERY " << query_selection.input_id << '\n';
  out << "REMARK HIKOBOSHI_INPUT_TARGET " << target_selection.input_id << '\n';
  out << "REMARK HIKOBOSHI_QUERY_SELECTION chain="
      << query_selection.chain_id << " model=" << query_selection.model_id
      << '\n';
  out << "REMARK HIKOBOSHI_TARGET_SELECTION chain="
      << target_selection.chain_id << " model=" << target_selection.model_id
      << '\n';
  api::AlignmentOptions alignment = options.alignment;
  if (api::is_package_default_gap(alignment.gap_open)) {
    alignment.gap_open = api::kDefaultGapOpen;
  }
  if (api::is_package_default_gap(alignment.gap_extension)) {
    alignment.gap_extension = api::kDefaultGapExtension;
  }
  out << "REMARK HIKOBOSHI_GAP_PARAMETERS open="
      << format_double(alignment.gap_open, options.metric_digits)
      << " extension="
      << format_double(alignment.gap_extension, options.metric_digits) << '\n';
  out << "REMARK HIKOBOSHI_BACKEND " << backend_name(options.backend) << '\n';
  out << "REMARK HIKOBOSHI_METRIC raw_sw_score "
      << format_double(result.metrics.raw_sw_score, options.metric_digits)
      << '\n';
  out << "REMARK HIKOBOSHI_METRIC aligned_pairs "
      << result.path.aligned_pairs << '\n';
  append_metric_remark(out, "coverage_query", result.metrics.coverage_query,
                       options.metric_digits);
  append_metric_remark(out, "coverage_target", result.metrics.coverage_target,
                       options.metric_digits);
  append_metric_remark(out, "coverage_mean", result.metrics.coverage_mean,
                       options.metric_digits);
  append_metric_remark(out, "identity", result.metrics.identity,
                       options.metric_digits);
  append_metric_remark(out, "rmsd", result.metrics.rmsd, options.metric_digits);
  append_metric_remark(out, "tm_score_query", result.metrics.tm_score_query,
                       options.metric_digits);
  append_metric_remark(out, "tm_score_target", result.metrics.tm_score_target,
                       options.metric_digits);
  append_metric_remark(out, "lddt", result.metrics.lddt, options.metric_digits);
  append_metric_remark(out, "lddt_byA", result.metrics.lddt_byA,
                       options.metric_digits);
  append_metric_remark(out, "lddt_byB", result.metrics.lddt_byB,
                       options.metric_digits);
  append_metric_remark(out, "lddt_aln", result.metrics.lddt_aln,
                       options.metric_digits);
  append_metric_remark(out, "coverage_byA", result.metrics.coverage_byA,
                       options.metric_digits);
  append_metric_remark(out, "coverage_byB", result.metrics.coverage_byB,
                       options.metric_digits);
  append_metric_remark(out, "ecs", result.metrics.ecs, options.metric_digits);
  out << "REMARK HIKOBOSHI_TRANSFORM target_to_query";
  if (!options.superpose_target) {
    out << " path_only";
  }
  out << " pairs=" << fit.pair_count;
  if (!fit.valid) {
    out << " invalid_reason=" << metric_invalid_reason(fit.reason);
  }
  out << '\n';
  append_transform_remark(out, fit.transform);
}

}  // namespace

universal::Result<PdbRenderResult> render_superposed_pdb(
    const api::PairwiseResult& result,
    const universal::StructureView& query,
    const universal::StructureView& target,
    const PdbWriterOptions& options) {
  if (!has_structure_coordinates(query) || !has_structure_coordinates(target)) {
    return {universal::unavailable_status("superposed PDB requires structure coordinates"), {}};
  }

  TransformFit fit{};
  if (options.superpose_target) {
    std::vector<Point3> query_points;
    std::vector<Point3> target_points;
    const universal::Status collect_status =
        collect_transform_points(result.path, query, target, query_points,
                                 target_points);
    if (!collect_status.ok()) {
      return {collect_status, {}};
    }
    fit = kabsch_superpose(query_points, target_points);
    if (!fit.valid) {
      PdbRenderResult invalid_result{};
      invalid_result.target_to_query = fit.transform;
      invalid_result.transform_valid = false;
      invalid_result.transform_invalid_reason = fit.reason;
      invalid_result.transform_pair_count = fit.pair_count;
      return {universal::unavailable_status("superposed PDB requires at least three observed CA pairs"),
              invalid_result};
    }
  } else {
    fit.valid = true;
    fit.reason = universal::MetricInvalidReason::None;
  }

  PdbRenderResult rendered{};
  rendered.target_to_query = fit.transform;
  rendered.transform_valid = fit.valid;
  rendered.transform_invalid_reason = fit.reason;
  rendered.transform_pair_count = fit.pair_count;

  std::ostringstream out;
  append_remarks(out, result, selection_from_structure(query, "query"),
                 selection_from_structure(target, "target"), options, fit);
  append_model(out, 1, query, PdbTransform{}, false, options);
  append_model(out, 2, target, fit.transform, options.superpose_target, options);
  out << "END\n";
  rendered.contents = out.str();
  return {universal::ok_status(), std::move(rendered)};
}

universal::Status write_superposed_pdb(
    std::string_view path,
    const api::PairwiseResult& result,
    const universal::StructureView& query,
    const universal::StructureView& target,
    const PdbWriterOptions& options) {
  universal::Result<PdbRenderResult> rendered =
      render_superposed_pdb(result, query, target, options);
  if (!rendered.status.ok()) {
    return rendered.status;
  }

  std::ofstream out{std::string{path}, std::ios::binary};
  if (!out) {
    return universal::unavailable_status("superposed PDB output path is not writable");
  }
  out << rendered.value.contents;
  if (!out) {
    return universal::unavailable_status("superposed PDB write failed");
  }
  return universal::ok_status();
}

}  // namespace hikoboshi::io
