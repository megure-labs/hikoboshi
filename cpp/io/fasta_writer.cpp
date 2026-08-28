#include <hikoboshi/io/fasta_writer.hpp>

#include <hikoboshi/io/all_vs_all_layout.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>

namespace hikoboshi::io {
namespace {

std::size_t metadata_residue_count(const FastaInputMetadata& metadata) {
  return metadata.residue_count == 0 ? metadata.residue_codes.size
                                     : metadata.residue_count;
}

universal::Status validate_sequence_metadata(
    const FastaInputMetadata& metadata) noexcept {
  if (!metadata.has_sequence_metadata) {
    return universal::unavailable_status("alignment FASTA requires sequence metadata");
  }
  const std::size_t residue_count =
      metadata.residue_count == 0 ? metadata.residue_codes.size
                                  : metadata.residue_count;
  if (residue_count > metadata.residue_codes.size ||
      (residue_count != 0 && metadata.residue_codes.data == nullptr)) {
    return universal::unavailable_status("alignment FASTA sequence metadata is incomplete");
  }
  return universal::ok_status();
}

universal::Status append_residue(const FastaInputMetadata& metadata,
                                 std::int32_t index,
                                 std::string& out) {
  if (index < 0) {
    return universal::invalid_argument_status("alignment path contains an invalid residue index");
  }
  const std::size_t residue_index = static_cast<std::size_t>(index);
  if (residue_index >= metadata_residue_count(metadata)) {
    return universal::invalid_argument_status("alignment path residue index is out of range");
  }
  out.push_back(metadata.residue_codes.data[residue_index]);
  return universal::ok_status();
}

std::string sanitized_header(std::string_view id, std::string_view fallback) {
  std::string header = sanitize_identifier(id.empty() ? fallback : id);
  if (header.empty()) {
    header = std::string{fallback};
  }
  return header;
}

void append_wrapped(std::string_view sequence,
                    std::size_t line_width,
                    std::string& out) {
  if (sequence.empty()) {
    out.push_back('\n');
    return;
  }
  for (std::size_t offset = 0; offset < sequence.size(); offset += line_width) {
    out.append(sequence.substr(offset, line_width));
    out.push_back('\n');
  }
}

}  // namespace

FastaInputMetadata fasta_metadata_from_structure(
    const universal::StructureView& structure) noexcept {
  FastaInputMetadata metadata{};
  metadata.input_id = structure.input_id;
  metadata.residue_codes = structure.residue_codes;
  metadata.residue_count = structure.residue_count;
  metadata.has_sequence_metadata =
      structure.residue_codes.data != nullptr &&
      structure.residue_codes.size >= structure.residue_count;
  return metadata;
}

universal::Result<GappedAlignmentFasta> alignment_path_to_gapped_sequences(
    const api::AlignmentPath& path,
    const FastaInputMetadata& query,
    const FastaInputMetadata& target) {
  universal::Status status = validate_sequence_metadata(query);
  if (!status.ok()) {
    return {status, {}};
  }
  status = validate_sequence_metadata(target);
  if (!status.ok()) {
    return {status, {}};
  }

  GappedAlignmentFasta gapped{};
  gapped.query.reserve(path.steps.size());
  gapped.target.reserve(path.steps.size());

  for (const auto& step : path.steps) {
    if (step.query_index < 0 && step.target_index < 0) {
      return {universal::invalid_argument_status("alignment path step cannot contain two gaps"),
              {}};
    }
    if (step.query_index >= 0) {
      status = append_residue(query, step.query_index, gapped.query);
      if (!status.ok()) {
        return {status, {}};
      }
    } else {
      gapped.query.push_back('-');
    }

    if (step.target_index >= 0) {
      status = append_residue(target, step.target_index, gapped.target);
      if (!status.ok()) {
        return {status, {}};
      }
    } else {
      gapped.target.push_back('-');
    }
  }

  return {universal::ok_status(), std::move(gapped)};
}

universal::Result<std::string> render_alignment_fasta(
    const api::AlignmentPath& path,
    const FastaInputMetadata& query,
    const FastaInputMetadata& target,
    const FastaWriterOptions& options) {
  if (options.line_width == 0) {
    return {universal::invalid_argument_status("FASTA line width must be non-zero"), {}};
  }

  universal::Result<GappedAlignmentFasta> gapped =
      alignment_path_to_gapped_sequences(path, query, target);
  if (!gapped.status.ok()) {
    return {gapped.status, {}};
  }

  std::string fasta;
  fasta.reserve(gapped.value.query.size() + gapped.value.target.size() +
                query.input_id.size() + target.input_id.size() + 16);
  fasta.push_back('>');
  fasta.append(sanitized_header(query.input_id, "query"));
  fasta.push_back('\n');
  append_wrapped(gapped.value.query, options.line_width, fasta);
  fasta.push_back('>');
  fasta.append(sanitized_header(target.input_id, "target"));
  fasta.push_back('\n');
  append_wrapped(gapped.value.target, options.line_width, fasta);
  return {universal::ok_status(), std::move(fasta)};
}

universal::Status write_alignment_fasta(
    std::string_view path,
    const api::AlignmentPath& alignment_path,
    const FastaInputMetadata& query,
    const FastaInputMetadata& target,
    const FastaWriterOptions& options) {
  universal::Result<std::string> rendered =
      render_alignment_fasta(alignment_path, query, target, options);
  if (!rendered.status.ok()) {
    return rendered.status;
  }

  std::ofstream out{std::string{path}, std::ios::binary};
  if (!out) {
    return universal::unavailable_status("alignment FASTA output path is not writable");
  }
  out << rendered.value;
  if (!out) {
    return universal::unavailable_status("alignment FASTA write failed");
  }
  return universal::ok_status();
}

}  // namespace hikoboshi::io
