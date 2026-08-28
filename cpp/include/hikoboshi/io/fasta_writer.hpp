#ifndef HIKOBOSHI_IO_FASTA_WRITER_HPP
#define HIKOBOSHI_IO_FASTA_WRITER_HPP

#include <cstddef>
#include <string>
#include <string_view>

#include <hikoboshi/api/results.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

struct FastaInputMetadata {
  std::string_view input_id{};
  universal::Span<const char> residue_codes{};
  std::size_t residue_count = 0;
  bool has_sequence_metadata = true;
};

struct FastaWriterOptions {
  std::size_t line_width = 80;
};

struct GappedAlignmentFasta {
  std::string query;
  std::string target;
};

FastaInputMetadata fasta_metadata_from_structure(
    const universal::StructureView& structure) noexcept;

[[nodiscard]] universal::Result<GappedAlignmentFasta>
alignment_path_to_gapped_sequences(
    const api::AlignmentPath& path,
    const FastaInputMetadata& query,
    const FastaInputMetadata& target);

[[nodiscard]] universal::Result<std::string> render_alignment_fasta(
    const api::AlignmentPath& path,
    const FastaInputMetadata& query,
    const FastaInputMetadata& target,
    const FastaWriterOptions& options = {});

[[nodiscard]] universal::Status write_alignment_fasta(
    std::string_view path,
    const api::AlignmentPath& alignment_path,
    const FastaInputMetadata& query,
    const FastaInputMetadata& target,
    const FastaWriterOptions& options = {});

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_FASTA_WRITER_HPP
