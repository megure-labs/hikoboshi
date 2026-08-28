#ifndef HIKOBOSHI_IO_DESIGN_FASTA_WRITER_HPP
#define HIKOBOSHI_IO_DESIGN_FASTA_WRITER_HPP

#include <string_view>

#include <hikoboshi/api/results.hpp>
#include <hikoboshi/universal/status.hpp>

namespace hikoboshi::io {

struct DesignFastaWriterOptions {
  float sampling_temp = 0.1F;
};

[[nodiscard]] universal::Status write_design_fasta(
    std::string_view path,
    const api::InverseFoldResult& result,
    const DesignFastaWriterOptions& options = {});

[[nodiscard]] universal::Status write_inverse_fold_logprobs_npz(
    std::string_view path,
    const api::InverseFoldResult& result);

}  // namespace hikoboshi::io

#endif  // HIKOBOSHI_IO_DESIGN_FASTA_WRITER_HPP
