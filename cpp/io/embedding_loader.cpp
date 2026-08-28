// Embedding loader: parses .npy little-endian float32 2D arrays into a
// LoadedEmbedding that owns the backing storage for an EmbeddingView. The
// charter requires no silent dtype coercion; non-float32 arrays are rejected.

#include <hikoboshi/io/embedding_loader.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <hikoboshi/universal/embedding.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {

LoadedEmbedding::LoadedEmbedding() : impl_(std::make_unique<Impl>()) {}
LoadedEmbedding::~LoadedEmbedding() = default;
LoadedEmbedding::LoadedEmbedding(LoadedEmbedding&&) noexcept = default;
LoadedEmbedding& LoadedEmbedding::operator=(LoadedEmbedding&&) noexcept =
    default;

universal::EmbeddingView LoadedEmbedding::view() const noexcept {
  universal::EmbeddingView v{};
  v.residue_count = impl_->residue_count;
  v.dimension = impl_->dimension;
  v.values =
      universal::Span<const float>{impl_->values.data(), impl_->values.size()};
  v.residue_codes = universal::Span<const char>{impl_->residue_codes.data(),
                                                impl_->residue_codes.size()};
  v.residues = universal::Span<const universal::ResidueMetadataView>{
      impl_->residues.data(), impl_->residues.size()};
  return v;
}

std::size_t LoadedEmbedding::residue_count() const noexcept {
  return impl_->residue_count;
}
std::size_t LoadedEmbedding::dimension() const noexcept {
  return impl_->dimension;
}
std::string_view LoadedEmbedding::input_id() const noexcept {
  return std::string_view{impl_->input_id_storage};
}
std::string_view LoadedEmbedding::source_filename() const noexcept {
  return std::string_view{impl_->source_filename_storage};
}
bool LoadedEmbedding::has_residue_metadata() const noexcept {
  return impl_->has_residue_metadata;
}

namespace {

bool starts_with(const std::uint8_t* data, std::size_t size, const char* magic,
                 std::size_t magic_size) noexcept {
  if (size < magic_size) return false;
  return std::equal(magic, magic + magic_size, data,
                    [](char lhs, std::uint8_t rhs) {
                      return static_cast<std::uint8_t>(lhs) == rhs;
                    });
}

std::string slice_to_string(const std::uint8_t* data, std::size_t size) {
  return std::string{reinterpret_cast<const char*>(data), size};
}

bool parse_npy_header(std::string_view header, char& dtype_byte_order,
                      char& dtype_kind,
                      std::size_t& dtype_size, bool& fortran_order,
                      std::vector<std::size_t>& shape, const char*& detail) {
  // Locate descr field.
  const std::size_t descr_pos = header.find("'descr'");
  const std::size_t fortran_pos = header.find("'fortran_order'");
  const std::size_t shape_pos = header.find("'shape'");
  if (descr_pos == std::string_view::npos ||
      fortran_pos == std::string_view::npos ||
      shape_pos == std::string_view::npos) {
    detail = "embedding npy header missing descr/fortran_order/shape";
    return false;
  }

  const std::size_t dtype_open = header.find('\'', descr_pos + 7);
  if (dtype_open == std::string_view::npos) {
    detail = "embedding npy descr token missing opening quote";
    return false;
  }
  const std::size_t dtype_close = header.find('\'', dtype_open + 1);
  if (dtype_close == std::string_view::npos) {
    detail = "embedding npy descr token missing closing quote";
    return false;
  }
  std::string descr{header.substr(dtype_open + 1, dtype_close - dtype_open - 1)};
  if (descr.empty()) {
    detail = "embedding npy descr empty";
    return false;
  }

  // Skip byte-order char (<, =, >, |).
  dtype_byte_order = '\0';
  std::size_t i = 0;
  if (descr[i] == '<' || descr[i] == '=' || descr[i] == '>' || descr[i] == '|') {
    dtype_byte_order = descr[i];
    ++i;
  }
  if (i >= descr.size()) {
    detail = "embedding npy descr missing kind";
    return false;
  }
  dtype_kind = descr[i++];
  std::string size_str = descr.substr(i);
  if (size_str.empty()) {
    detail = "embedding npy descr missing size";
    return false;
  }
  dtype_size = 0;
  const auto dtype_size_parse =
      std::from_chars(size_str.data(), size_str.data() + size_str.size(),
                      dtype_size);
  if (dtype_size_parse.ec != std::errc{} ||
      dtype_size_parse.ptr != size_str.data() + size_str.size()) {
    detail = "embedding npy descr has non-numeric size";
    return false;
  }

  const std::size_t fortran_colon = header.find(':', fortran_pos);
  if (fortran_colon != std::string_view::npos) {
    std::size_t j = fortran_colon + 1;
    while (j < header.size() && std::isspace(static_cast<unsigned char>(header[j]))) {
      ++j;
    }
    fortran_order = header.compare(j, 4, "True") == 0;
  } else {
    fortran_order = false;
  }

  const std::size_t shape_open = header.find('(', shape_pos);
  if (shape_open == std::string_view::npos) {
    detail = "embedding npy shape tuple missing opening paren";
    return false;
  }
  const std::size_t shape_close = header.find(')', shape_open);
  if (shape_close == std::string_view::npos) {
    detail = "embedding npy shape tuple missing closing paren";
    return false;
  }
  std::string_view shape_str =
      header.substr(shape_open + 1, shape_close - shape_open - 1);
  shape.clear();
  std::size_t cursor = 0;
  while (cursor < shape_str.size()) {
    while (cursor < shape_str.size() &&
           (std::isspace(static_cast<unsigned char>(shape_str[cursor])) ||
            shape_str[cursor] == ',')) {
      ++cursor;
    }
    if (cursor >= shape_str.size()) break;
    std::size_t start = cursor;
    while (cursor < shape_str.size() && shape_str[cursor] >= '0' &&
           shape_str[cursor] <= '9') {
      ++cursor;
    }
    if (start == cursor) {
      break;
    }
    std::size_t value = 0;
    const std::string_view token = shape_str.substr(start, cursor - start);
    const auto shape_parse =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (shape_parse.ec != std::errc{} ||
        shape_parse.ptr != token.data() + token.size()) {
      detail = "embedding npy shape tuple has non-numeric dimension";
      return false;
    }
    shape.push_back(value);
  }
  return true;
}

}  // namespace

universal::Result<LoadedEmbedding> load_embedding_from_npy_bytes(
    universal::Span<const std::uint8_t> bytes,
    std::string_view origin_label,
    const EmbeddingLoadOptions& options) {
  universal::Result<LoadedEmbedding> result{};
  result.value = LoadedEmbedding{};
  auto& impl = result.value.impl();
  impl.source_filename_storage = std::string{origin_label};
  impl.input_id_storage = std::string{origin_label};

  if (bytes.size < 10 ||
      !starts_with(bytes.data, bytes.size, "\x93NUMPY", 6)) {
    result.status = universal::invalid_argument_status(
        "embedding input is not a NumPy .npy file");
    return result;
  }

  const std::uint8_t major = bytes.data[6];
  const std::uint8_t minor = bytes.data[7];
  std::size_t header_len = 0;
  std::size_t header_offset = 0;
  if (major == 1) {
    header_len = static_cast<std::size_t>(bytes.data[8]) |
                 (static_cast<std::size_t>(bytes.data[9]) << 8);
    header_offset = 10;
  } else if (major == 2 || major == 3) {
    if (bytes.size < 12) {
      result.status =
          universal::invalid_argument_status("embedding npy header truncated");
      return result;
    }
    header_len = static_cast<std::size_t>(bytes.data[8]) |
                 (static_cast<std::size_t>(bytes.data[9]) << 8) |
                 (static_cast<std::size_t>(bytes.data[10]) << 16) |
                 (static_cast<std::size_t>(bytes.data[11]) << 24);
    header_offset = 12;
  } else {
    (void)minor;
    result.status =
        universal::invalid_argument_status("embedding npy version unsupported");
    return result;
  }

  if (bytes.size < header_offset + header_len) {
    result.status =
        universal::invalid_argument_status("embedding npy header runs past file");
    return result;
  }
  const std::string header =
      slice_to_string(bytes.data + header_offset, header_len);

  char dtype_byte_order = '\0';
  char dtype_kind = '\0';
  std::size_t dtype_size = 0;
  bool fortran_order = false;
  std::vector<std::size_t> shape;
  const char* detail = "";
  if (!parse_npy_header(header, dtype_byte_order, dtype_kind, dtype_size,
                        fortran_order, shape, detail)) {
    result.status = universal::invalid_argument_status(detail);
    return result;
  }
  if (fortran_order) {
    result.status = universal::invalid_argument_status(
        "embedding npy is fortran-order; only C-order arrays are accepted");
    return result;
  }
  if (dtype_byte_order != '<' || dtype_kind != 'f' || dtype_size != 4) {
    result.status = universal::invalid_argument_status(
        "embedding npy dtype must be little-endian float32; refusing silent "
        "coercion");
    return result;
  }
  if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
    result.status = universal::invalid_argument_status(
        "embedding npy shape must be 2D (residue_count, dimension)");
    return result;
  }
  if (options.expected_dimension.has_value() &&
      shape[1] != *options.expected_dimension) {
    result.status = universal::invalid_argument_status(
        "embedding npy dimension does not match expected");
    return result;
  }
  if (options.expected_residue_count.has_value() &&
      shape[0] != *options.expected_residue_count) {
    result.status = universal::invalid_argument_status(
        "embedding npy residue count does not match expected");
    return result;
  }

  if (shape[0] >
      std::numeric_limits<std::size_t>::max() / shape[1] / sizeof(float)) {
    result.status = universal::invalid_argument_status(
        "embedding npy shape byte size overflows size_t");
    return result;
  }
  const std::size_t expected_bytes = shape[0] * shape[1] * sizeof(float);
  const std::size_t data_offset = header_offset + header_len;
  if (bytes.size < data_offset + expected_bytes) {
    result.status =
        universal::invalid_argument_status("embedding npy payload truncated");
    return result;
  }

  impl.residue_count = shape[0];
  impl.dimension = shape[1];
  impl.values.resize(shape[0] * shape[1]);
  std::memcpy(impl.values.data(), bytes.data + data_offset, expected_bytes);

  result.status = universal::ok_status();
  return result;
}

universal::Result<LoadedEmbedding> load_embedding_from_file(
    std::string_view path,
    const EmbeddingLoadOptions& options) {
  universal::Result<LoadedEmbedding> result{};
  result.value = LoadedEmbedding{};

  std::ifstream stream{std::string(path), std::ios::binary};
  if (!stream) {
    result.status =
        universal::unavailable_status("embedding file not readable");
    return result;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  const std::string content = buffer.str();
  const auto* bytes_ptr =
      reinterpret_cast<const std::uint8_t*>(content.data());
  return load_embedding_from_npy_bytes(
      universal::Span<const std::uint8_t>{bytes_ptr, content.size()},
      path, options);
}

void attach_residue_metadata(LoadedEmbedding& embedding,
                             universal::Span<const char> residue_codes) {
  auto& impl = embedding.impl();
  if (residue_codes.size != impl.residue_count) {
    return;
  }
  impl.residue_codes.assign(residue_codes.data,
                            residue_codes.data + residue_codes.size);
  impl.residues.clear();
  impl.residues.reserve(impl.residue_count);
  for (std::size_t i = 0; i < impl.residue_count; ++i) {
    universal::ResidueMetadataView md{};
    md.residue_code = residue_codes.data[i];
    md.original_residue_name = std::string_view{};
    md.chain_id = std::string_view{};
    md.model_id = std::string_view{};
    md.model_index = 0;
    md.residue_number = static_cast<std::int32_t>(i + 1);
    md.insertion_code = ' ';
    md.source_id = std::string_view{};
    md.source_residue_index = static_cast<std::int64_t>(i);
    md.source_filename = std::string_view{impl.source_filename_storage};
    md.source_record_index = -1;
    impl.residues.push_back(md);
  }
  impl.has_residue_metadata = true;
}

}  // namespace hikoboshi::io
