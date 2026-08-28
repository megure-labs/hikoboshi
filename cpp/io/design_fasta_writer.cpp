#include <hikoboshi/io/design_fasta_writer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace hikoboshi::io {
namespace {

constexpr universal::Status ok() noexcept { return universal::ok_status(); }

constexpr universal::Status invalid(const char* detail) noexcept {
  return universal::invalid_argument_status(detail);
}

constexpr universal::Status unavailable(const char* detail) noexcept {
  return universal::unavailable_status(detail);
}

const char* decode_order_name(api::InverseFoldDecodeOrder order) noexcept {
  switch (order) {
    case api::InverseFoldDecodeOrder::Random:
      return "random";
    case api::InverseFoldDecodeOrder::NToC:
      return "n_to_c";
  }
  return "random";
}

void append_u16_le(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

std::uint32_t crc32(const std::vector<std::uint8_t>& bytes) {
  std::uint32_t crc = 0xFFFFFFFFU;
  for (const std::uint8_t byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xEDB88320U & mask);
    }
  }
  return crc ^ 0xFFFFFFFFU;
}

std::vector<std::uint8_t> make_npy_payload(
    const api::InverseFoldLogProbsArtifact& artifact) {
  std::vector<std::uint8_t> out;
  out.push_back(0x93U);
  out.insert(out.end(), {'N', 'U', 'M', 'P', 'Y'});
  out.push_back(1U);
  out.push_back(0U);

  std::ostringstream header_stream;
  header_stream << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                << artifact.num_seqs << ", " << artifact.residue_count
                << ", " << artifact.vocab_size << "), }";
  std::string header = header_stream.str();
  const std::size_t padding =
      (16U - ((10U + header.size() + 1U) % 16U)) % 16U;
  header.append(padding, ' ');
  header.push_back('\n');
  append_u16_le(out, static_cast<std::uint16_t>(header.size()));
  out.insert(out.end(), header.begin(), header.end());

  out.reserve(out.size() + artifact.values.size() * sizeof(float));
  for (const float value : artifact.values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    append_u32_le(out, bits);
  }
  return out;
}

bool fits_u32(std::size_t value) noexcept {
  return value <= static_cast<std::size_t>(
                      std::numeric_limits<std::uint32_t>::max());
}

universal::Status write_npz_file(std::string_view path,
                                 const std::vector<std::uint8_t>& payload) {
  constexpr std::string_view kName = "log_probs.npy";
  if (!fits_u32(payload.size())) {
    return invalid("inverse-fold log-prob artifact is too large for ZIP32");
  }
  const std::uint32_t payload_size =
      static_cast<std::uint32_t>(payload.size());
  const std::uint32_t crc = crc32(payload);
  std::vector<std::uint8_t> zip;

  const std::uint32_t local_header_offset = 0;
  append_u32_le(zip, 0x04034B50U);
  append_u16_le(zip, 20U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u32_le(zip, crc);
  append_u32_le(zip, payload_size);
  append_u32_le(zip, payload_size);
  append_u16_le(zip, static_cast<std::uint16_t>(kName.size()));
  append_u16_le(zip, 0U);
  zip.insert(zip.end(), kName.begin(), kName.end());
  zip.insert(zip.end(), payload.begin(), payload.end());

  if (!fits_u32(zip.size())) {
    return invalid("inverse-fold log-prob artifact is too large for ZIP32");
  }
  const std::uint32_t central_directory_offset =
      static_cast<std::uint32_t>(zip.size());
  append_u32_le(zip, 0x02014B50U);
  append_u16_le(zip, 20U);
  append_u16_le(zip, 20U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u32_le(zip, crc);
  append_u32_le(zip, payload_size);
  append_u32_le(zip, payload_size);
  append_u16_le(zip, static_cast<std::uint16_t>(kName.size()));
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u32_le(zip, 0U);
  append_u32_le(zip, local_header_offset);
  zip.insert(zip.end(), kName.begin(), kName.end());

  const std::uint32_t central_directory_size =
      static_cast<std::uint32_t>(zip.size()) - central_directory_offset;
  append_u32_le(zip, 0x06054B50U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 0U);
  append_u16_le(zip, 1U);
  append_u16_le(zip, 1U);
  append_u32_le(zip, central_directory_size);
  append_u32_le(zip, central_directory_offset);
  append_u16_le(zip, 0U);

  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return unavailable("inverse-fold log-prob artifact is not writable");
  }
  out.write(reinterpret_cast<const char*>(zip.data()),
            static_cast<std::streamsize>(zip.size()));
  if (!out) {
    return unavailable("inverse-fold log-prob artifact could not be written");
  }
  return ok();
}

}  // namespace

universal::Status write_design_fasta(
    std::string_view path,
    const api::InverseFoldResult& result,
    const DesignFastaWriterOptions& options) {
  if (path.empty()) {
    return invalid("design FASTA output path is required");
  }
  std::ofstream out{std::string(path)};
  if (!out) {
    return unavailable("design FASTA output is not writable");
  }
  for (std::size_t index = 0; index < result.sequences.size(); ++index) {
    const api::InverseFoldSequenceResult& sequence = result.sequences[index];
    out << ">design_" << (index + 1U) << " score=" << std::setprecision(7)
        << sequence.score << " seed=" << sequence.seed << " T="
        << options.sampling_temp << " decode_order="
        << decode_order_name(sequence.decode_order) << "\n";
    for (std::size_t offset = 0; offset < sequence.sequence.size();
         offset += 80U) {
      const std::size_t chunk =
          std::min<std::size_t>(80U, sequence.sequence.size() - offset);
      out.write(sequence.sequence.data() + offset,
                static_cast<std::streamsize>(chunk));
      out << "\n";
    }
  }
  if (!out) {
    return unavailable("design FASTA output could not be written");
  }
  return ok();
}

universal::Status write_inverse_fold_logprobs_npz(
    std::string_view path,
    const api::InverseFoldResult& result) {
  if (path.empty()) {
    return invalid("inverse-fold log-prob output path is required");
  }
  const api::InverseFoldLogProbsArtifact& artifact = result.logprobs;
  const std::size_t expected = artifact.num_seqs * artifact.residue_count *
                               artifact.vocab_size;
  if (artifact.values.size() != expected || expected == 0) {
    return invalid("inverse-fold log-prob artifact shape is invalid");
  }
  std::vector<std::uint8_t> payload = make_npy_payload(artifact);
  return write_npz_file(path, payload);
}

}  // namespace hikoboshi::io
