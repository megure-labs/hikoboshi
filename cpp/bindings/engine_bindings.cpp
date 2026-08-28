#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/bindings/args.hpp>
#include <hikoboshi/bindings/casters.hpp>
#include <hikoboshi/bindings/dict_builder.hpp>
#include <hikoboshi/bindings/engine_helpers.hpp>
#include <hikoboshi/bindings/list_builder.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>
#include <hikoboshi/bindings/span_conversion.hpp>
#include <hikoboshi/io/structure_loader.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace hikoboshi::bindings {

PyObject* encode_result_to_python(const api::EncodeResult& result);
PyObject* pairwise_result_to_python(const api::PairwiseResult& result);

namespace {

struct StructureOptionFields {
  std::string chain_id;
  PyObject* chain_index = Py_None;
  std::string pdb_model_id;
  PyObject* pdb_model_index = Py_None;
};

bool parse_optional_string_field(TypedArgParser& parser,
                                 const char* name,
                                 std::string& out) {
  PyObject* object = Py_None;
  return parser.optional_object(name, object, Py_None) &&
         optional_string_arg(object, name, out);
}

// Tokenize an AA letter string against the hikoboshi-esm2-8m compacted
// 29-row vocab. The Python bindings cannot include the weights `detail`
// surface, so the token mapping is inlined here. Drift between this
// table and `cpp/weights/embedded_esm2_8m.cpp:kEsm2TokenTable` is caught
// at review time.
std::int32_t esm2_aa_token(char c) noexcept {
  static constexpr struct {
    char letter;
    std::int32_t token_id;
  } kTokens[] = {
      // The embedded weight rows are indexed by the training dataset's
      // aatype, which the encoder passes through unchanged
      // (esm2_encoder.forward: `tokens = aatype.clamp(0, 19)`). Recovering
      // aatype from SCOPe40-test-v2's proteins.h5 against the residue letters
      // gives a bijection at purity 1.0000 over ~263k residues: the order is
      // ALPHABETICAL by one-letter code. NOT the AlphaFold-style
      // ARNDCQEGHILKMFPSTWYV this table used to carry, and not FAIR ESM-2's
      // frequency order. PAD=25, CLS/BOS=26, EOS=27, MASK=28 follow.
      //
      // aatype is strictly 0-19 in the training data, so rows 20-24 were
      // never trained; non-standard letters clamp to 19 as training did
      // rather than addressing untrained rows.
      {'A', 0},  {'C', 1},  {'D', 2},  {'E', 3},  {'F', 4},  {'G', 5},
      {'H', 6},  {'I', 7},  {'K', 8},  {'L', 9},  {'M', 10}, {'N', 11},
      {'P', 12}, {'Q', 13}, {'R', 14}, {'S', 15}, {'T', 16}, {'V', 17},
      {'W', 18}, {'Y', 19}, {'B', 19}, {'U', 19}, {'Z', 19}, {'O', 19},
      {'X', 19},
  };
  for (const auto& entry : kTokens) {
    if (entry.letter == c) {
      return entry.token_id;
    }
  }
  return -1;
}

bool tokenize_python_sequence(PyObject* sequence_arg,
                              const char* label,
                              std::vector<std::int32_t>& tokens) {
  tokens.clear();
  if (sequence_arg == nullptr || !PyUnicode_Check(sequence_arg)) {
    PyErr_Format(PyExc_TypeError, "%s must be a string", label);
    return false;
  }
  Py_ssize_t size = 0;
  const char* text = PyUnicode_AsUTF8AndSize(sequence_arg, &size);
  if (text == nullptr) {
    return false;
  }
  tokens.reserve(static_cast<std::size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    const char c = text[i];
    if (c == '-' || c == '*') continue;
    const char upper = (c >= 'a' && c <= 'z')
                           ? static_cast<char>(c - 'a' + 'A')
                           : c;
    std::int32_t token = esm2_aa_token(upper);
    if (token < 0) {
      token = 24;  // Map unknown AA codes to 'X'.
    }
    tokens.push_back(token);
  }
  if (tokens.empty()) {
    raise_if_error(invalid_argument(
        "sequence input must contain at least one valid AA residue"));
    return false;
  }
  return true;
}

bool parse_pairwise_mode_kwargs(TypedArgParser& parser,
                                api::AlignmentMode& mode,
                                double& temperature) {
  std::string mode_text = "hard";
  if (!parse_optional_string_field(parser, "mode", mode_text)) {
    return false;
  }
  if (mode_text.empty()) {
    mode_text = "hard";
  }
  if (!parser.optional("temperature", temperature,
                       static_cast<double>(api::kDefaultSoftTemperature))) {
    return false;
  }
  if (mode_text == "soft") {
    mode = api::AlignmentMode::Soft;
  } else if (mode_text == "hard") {
    mode = api::AlignmentMode::Hard;
  } else if (mode_text == "both") {
    mode = api::AlignmentMode::Both;
  } else {
    raise_if_error(invalid_argument(
        "pairwise mode must be 'hard', 'soft', or 'both'"));
    return false;
  }
  return true;
}

bool parse_structure_options(TypedArgParser& parser,
                             StructureOptionFields& fields) {
  return parse_optional_string_field(parser, "chain_id", fields.chain_id) &&
         parser.optional_object("chain_index", fields.chain_index, Py_None) &&
         parse_optional_string_field(parser, "pdb_model_id",
                                     fields.pdb_model_id) &&
         parser.optional_object("pdb_model_index", fields.pdb_model_index,
                                Py_None);
}

int apply_structure_options(const StructureOptionFields& fields,
                            io::StructureLoadOptions& options) {
  return bindings::apply_structure_options(
      fields.chain_index, fields.chain_id, fields.pdb_model_id,
      fields.pdb_model_index, options);
}

class ScopedGilRelease final {
 public:
  ScopedGilRelease() : state_(PyEval_SaveThread()) {}
  ScopedGilRelease(const ScopedGilRelease&) = delete;
  ScopedGilRelease& operator=(const ScopedGilRelease&) = delete;
  ~ScopedGilRelease() { PyEval_RestoreThread(state_); }

 private:
  PyThreadState* state_;
};

const char* dynamic_status_detail(const std::string& message) {
  thread_local std::string buffer;
  buffer = message;
  return buffer.c_str();
}

universal::Status dynamic_invalid_argument(const std::string& message) {
  return universal::invalid_argument_status(dynamic_status_detail(message));
}

bool python_string_to_std(PyObject* object,
                          const char* label,
                          std::string& out) {
  if (!PyUnicode_Check(object)) {
    PyErr_Format(PyExc_TypeError, "%s must be a string", label);
    return false;
  }
  Py_ssize_t size = 0;
  const char* text = PyUnicode_AsUTF8AndSize(object, &size);
  if (text == nullptr) {
    return false;
  }
  out.assign(text, static_cast<std::size_t>(size));
  return true;
}

bool parse_pair_list_pairs(
    PyObject* pairs_arg,
    std::vector<std::pair<std::string, std::string>>& pairs) {
  PyObjectRef sequence(
      PySequence_Fast(pairs_arg, "pairs must be a sequence"));
  if (!sequence) {
    return false;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());
  pairs.clear();
  pairs.reserve(static_cast<std::size_t>(count));
  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < count; ++index) {
    PyObjectRef pair(PySequence_Fast(
        items[index], "each pair must be a two-item sequence"));
    if (!pair) {
      return false;
    }
    if (PySequence_Fast_GET_SIZE(pair.get()) != 2) {
      raise_if_error(dynamic_invalid_argument(
          "pairs[" + std::to_string(static_cast<long long>(index)) +
          "] must contain exactly two strings"));
      return false;
    }
    PyObject** pair_items = PySequence_Fast_ITEMS(pair.get());
    std::string query;
    std::string target;
    if (!python_string_to_std(pair_items[0], "query_id", query) ||
        !python_string_to_std(pair_items[1], "target_id", target)) {
      return false;
    }
    pairs.emplace_back(std::move(query), std::move(target));
  }
  return true;
}

bool parse_pair_list_mode_kwargs(TypedArgParser& parser,
                                 api::AlignmentMode& mode,
                                 double& temperature) {
  std::string mode_text = "hard";
  if (!parse_optional_string_field(parser, "mode", mode_text)) {
    return false;
  }
  if (mode_text.empty()) {
    mode_text = "hard";
  }
  if (!parser.optional("temperature", temperature,
                       static_cast<double>(api::kDefaultSoftTemperature))) {
    return false;
  }
  if (mode_text == "soft") {
    mode = api::AlignmentMode::Soft;
  } else if (mode_text == "hard") {
    mode = api::AlignmentMode::Hard;
  } else if (mode_text == "both") {
    mode = api::AlignmentMode::Both;
  } else {
    raise_if_error(invalid_argument(
        "pair-list mode must be 'hard', 'soft', or 'both'"));
    return false;
  }
  return true;
}

bool parse_pair_list_thread_kwargs(PyObject* threads_arg,
                                   PyObject* thread_count_arg,
                                   std::uint32_t& thread_count) {
  if (threads_arg != nullptr && thread_count_arg != nullptr) {
    raise_if_error(
        invalid_argument("use either threads or thread_count, not both"));
    return false;
  }
  PyObject* selected = threads_arg != nullptr ? threads_arg : thread_count_arg;
  return parse_thread_count_arg(selected, thread_count);
}

template <typename Request>
universal::Result<api::AllVsAllResult> collect_pair_list_without_gil(
    const api::Engine& engine,
    const Request& request) {
  ScopedGilRelease gil_release;
  return engine.collect_pair_list(request);
}

PyObject* pair_list_result_to_python(
    const api::AllVsAllResult& result) {
  ListBuilder list(result.records.size());
  if (!list) {
    return nullptr;
  }
  for (std::size_t index = 0; index < result.records.size(); ++index) {
    if (!list.set_new_ref(
            index, pairwise_result_to_python(result.records[index].result))) {
      return nullptr;
    }
  }
  return list.release();
}

std::vector<std::int32_t> tokenize_sequence_text(std::string_view sequence) {
  std::vector<std::int32_t> tokens;
  tokens.reserve(sequence.size());
  for (char c : sequence) {
    if (c == '-' || c == '*') continue;
    const char upper = (c >= 'a' && c <= 'z')
                           ? static_cast<char>(c - 'a' + 'A')
                           : c;
    std::int32_t token = esm2_aa_token(upper);
    if (token < 0) {
      token = 24;
    }
    tokens.push_back(token);
  }
  return tokens;
}

struct FastaRecords {
  std::vector<std::string> names;
  std::vector<std::vector<std::int32_t>> tokens;
};

universal::Status read_fasta_records(const std::string& path,
                                     FastaRecords& records) {
  std::ifstream in(path);
  if (!in) {
    return universal::unavailable_status(
        "sequence pair-list input FASTA is not readable");
  }

  records.names.clear();
  records.tokens.clear();
  std::vector<std::string> sequences;
  std::string line;
  std::string current_name;
  std::string current_sequence;
  const auto flush_record = [&]() {
    if (current_name.empty() && current_sequence.empty()) return;
    records.names.push_back(
        current_name.empty()
            ? std::string{"seq" + std::to_string(records.names.size())}
            : current_name);
    sequences.push_back(current_sequence);
    current_name.clear();
    current_sequence.clear();
  };

  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) continue;
    if (line.front() == '>') {
      flush_record();
      current_name = line.substr(1);
      while (!current_name.empty() &&
             (current_name.back() == ' ' || current_name.back() == '\t')) {
        current_name.pop_back();
      }
      continue;
    }
    for (char c : line) {
      if (c == ' ' || c == '\t') continue;
      current_sequence.push_back(c);
    }
  }
  flush_record();

  if (!in.eof()) {
    return universal::unavailable_status(
        "sequence pair-list input FASTA could not be read");
  }
  if (sequences.empty()) {
    return universal::invalid_argument_status(
        "sequence pair-list FASTA contained no records");
  }

  records.tokens.reserve(sequences.size());
  for (const std::string& sequence : sequences) {
    std::vector<std::int32_t> tokens = tokenize_sequence_text(sequence);
    if (tokens.empty()) {
      return universal::invalid_argument_status(
          "sequence pair-list FASTA contains a record with no valid AA residues");
    }
    records.tokens.push_back(std::move(tokens));
  }
  return universal::ok_status();
}

universal::Result<api::Engine> make_sequence_package_engine(
    PyObject* package_object,
    std::uint32_t thread_count) {
  auto package =
      (package_object == nullptr || package_object == Py_None)
          ? weights::default_package(weights::kDefaultEsm2_8mModelName)
          : package_from_python(package_object);
  if (!status_ok(package.status)) {
    return {package.status, api::Engine{}};
  }
  if (package.value.descriptor == nullptr) {
    return {failed_precondition(
                "default Hikoboshi sequence package descriptor is missing"),
            api::Engine{}};
  }

  api::EngineConfig config{};
  config.package = package.value;
  config.weights = package.value.descriptor->compatibility_views.weights;
  config.execution.thread_count = thread_count;
  return {universal::ok_status(), api::Engine{config}};
}

bool ascii_case_equal(std::string_view lhs, std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    char left = lhs[index];
    char right = rhs[index];
    if (left >= 'A' && left <= 'Z') {
      left = static_cast<char>(left - 'A' + 'a');
    }
    if (right >= 'A' && right <= 'Z') {
      right = static_cast<char>(right - 'A' + 'a');
    }
    if (left != right) {
      return false;
    }
  }
  return true;
}

bool ends_with_ascii_case(std::string_view value,
                          std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         ascii_case_equal(value.substr(value.size() - suffix.size()), suffix);
}

bool is_structure_path(const std::filesystem::path& path) {
  const std::string filename = path.filename().string();
  return ends_with_ascii_case(filename, ".pdb") ||
         ends_with_ascii_case(filename, ".ent") ||
         ends_with_ascii_case(filename, ".cif") ||
         ends_with_ascii_case(filename, ".mmcif");
}

universal::Status list_structure_files(
    const std::string& directory,
    std::vector<std::filesystem::path>& files) {
  std::error_code ec;
  if (!std::filesystem::is_directory(directory, ec) || ec) {
    return universal::unavailable_status(
        "pair-list PDB directory is not readable");
  }
  files.clear();
  for (std::filesystem::directory_iterator it{directory, ec};
       !ec && it != std::filesystem::directory_iterator{}; it.increment(ec)) {
    std::error_code file_ec;
    if (!it->is_regular_file(file_ec) || file_ec) {
      continue;
    }
    if (is_structure_path(it->path())) {
      files.push_back(it->path());
    }
  }
  if (ec) {
    return universal::unavailable_status(
        "pair-list PDB directory could not be read");
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    return universal::invalid_argument_status(
        "pair-list PDB directory contained no PDB or mmCIF files");
  }
  return universal::ok_status();
}

bool load_structures_from_directory(
    PyObject* directory_arg,
    const io::StructureLoadOptions& options,
    std::vector<io::LoadedStructure>& loaded,
    std::vector<universal::StructureView>& views) {
  std::string directory;
  if (!path_from_python(directory_arg, directory)) {
    return false;
  }
  std::vector<std::filesystem::path> files;
  universal::Status status = list_structure_files(directory, files);
  if (raise_if_error(status) < 0) {
    return false;
  }

  loaded.clear();
  views.clear();
  loaded.reserve(files.size());
  views.reserve(files.size());
  for (const std::filesystem::path& file : files) {
    universal::Result<io::LoadedStructure> structure =
        io::load_structure_from_file(file.string(), options);
    if (raise_if_error(structure.status) < 0) {
      return false;
    }
    loaded.push_back(std::move(structure.value));
  }
  for (const io::LoadedStructure& structure : loaded) {
    views.push_back(structure.view());
  }
  return true;
}

bool load_structure_sequence(PyObject* inputs_arg,
                             const io::StructureLoadOptions& options,
                             std::vector<io::LoadedStructure>& loaded,
                             std::vector<universal::StructureView>& views) {
  PyObjectRef sequence(
      PySequence_Fast(inputs_arg, "structure inputs must be a sequence"));
  if (!sequence) {
    return false;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());
  loaded.clear();
  views.clear();
  loaded.reserve(static_cast<std::size_t>(count));
  views.reserve(static_cast<std::size_t>(count));

  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < count; ++index) {
    io::LoadedStructure structure;
    if (!load_structure_arg(items[index], options, structure)) {
      return false;
    }
    loaded.push_back(std::move(structure));
  }
  for (const io::LoadedStructure& structure : loaded) {
    views.push_back(structure.view());
  }
  return true;
}

PyObjectRef metadata_for_index(PyObject* metadata, Py_ssize_t index) {
  if (metadata == nullptr || metadata == Py_None) {
    return PyObjectRef::borrow(Py_None);
  }
  PyObjectRef sequence(PySequence_Fast(metadata, "metadata must be a sequence"));
  if (!sequence) {
    return {};
  }
  const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.get());
  if (index >= size) {
    PyErr_SetString(PyExc_ValueError,
                    "metadata length must match embedding input count");
    return {};
  }
  return PyObjectRef::borrow(PySequence_Fast_GET_ITEM(sequence.get(), index));
}

class NamedEmbeddingInput final {
 public:
  bool acquire(PyObject* object,
               PyObject* metadata,
               const char* label) {
    if (!buffer_.acquire(object, label, BufferContiguity::C)) {
      raise_current_error_as_invalid_argument(
          "embedding input must expose a Python buffer");
      return false;
    }
    if (buffer_.extent(0) <= 0 || buffer_.extent(1) <= 0) {
      raise_if_error(
          invalid_argument("embedding input must be a non-empty 2D array"));
      return false;
    }
    const std::size_t residue_count =
        static_cast<std::size_t>(buffer_.extent(0));
    if (!metadata_fields(metadata, residue_count)) {
      raise_current_error_as_invalid_argument("embedding metadata is invalid");
      return false;
    }
    return true;
  }

  universal::EmbeddingView view() const noexcept {
    const Py_buffer& py_buffer = buffer_.py_buffer();
    return {static_cast<std::size_t>(py_buffer.shape[0]),
            static_cast<std::size_t>(py_buffer.shape[1]),
            {static_cast<const float*>(py_buffer.buf),
             static_cast<std::size_t>(py_buffer.shape[0] *
                                      py_buffer.shape[1])},
            {residue_codes_.empty() ? nullptr : residue_codes_.data(),
             residue_codes_.size()},
            {residues_.empty() ? nullptr : residues_.data(),
             residues_.size()}};
  }

 private:
  bool metadata_fields(PyObject* metadata, std::size_t residue_count) {
    residue_codes_.clear();
    residues_.clear();
    source_id_.clear();
    source_filename_.clear();
    if (metadata == nullptr || metadata == Py_None) {
      return true;
    }
    if (!PyDict_Check(metadata)) {
      PyErr_SetString(PyExc_TypeError, "metadata must be a dict or None");
      return false;
    }

    PyObject* codes = PyDict_GetItemString(metadata, "residue_codes");
    if (codes != nullptr && codes != Py_None &&
        !metadata_residue_codes(codes, residue_count)) {
      return false;
    }
    PyObject* source = PyDict_GetItemString(metadata, "source_id");
    if (source == nullptr || source == Py_None) {
      source = PyDict_GetItemString(metadata, "input_id");
    }
    if (source != nullptr && source != Py_None &&
        !python_string_to_std(source, "metadata source_id", source_id_)) {
      return false;
    }
    PyObject* filename = PyDict_GetItemString(metadata, "source_filename");
    if (filename != nullptr && filename != Py_None &&
        !python_string_to_std(filename, "metadata source_filename",
                              source_filename_)) {
      return false;
    }
    if (source_filename_.empty()) {
      source_filename_ = source_id_;
    }

    if (!source_id_.empty()) {
      residues_.resize(residue_count);
      for (std::size_t index = 0; index < residue_count; ++index) {
        universal::ResidueMetadataView md{};
        md.residue_code =
            residue_codes_.empty() ? 'X' : residue_codes_[index];
        md.model_index = 0;
        md.residue_number = static_cast<std::int32_t>(index + 1U);
        md.insertion_code = ' ';
        md.source_id = source_id_;
        md.source_residue_index = static_cast<std::int64_t>(index);
        md.source_filename = source_filename_;
        md.source_record_index = -1;
        residues_[index] = md;
      }
    }
    return true;
  }

  bool metadata_residue_codes(PyObject* codes, std::size_t residue_count) {
    if (PyUnicode_Check(codes)) {
      Py_ssize_t size = 0;
      const char* text = PyUnicode_AsUTF8AndSize(codes, &size);
      if (text == nullptr) {
        return false;
      }
      if (static_cast<std::size_t>(size) != residue_count) {
        PyErr_SetString(PyExc_ValueError,
                        "metadata residue_codes length must match embeddings");
        return false;
      }
      residue_codes_.assign(text, text + size);
      return true;
    }
    if (PyBytes_Check(codes)) {
      char* text = nullptr;
      Py_ssize_t size = 0;
      if (PyBytes_AsStringAndSize(codes, &text, &size) < 0) {
        return false;
      }
      if (static_cast<std::size_t>(size) != residue_count) {
        PyErr_SetString(PyExc_ValueError,
                        "metadata residue_codes length must match embeddings");
        return false;
      }
      residue_codes_.assign(text, text + size);
      return true;
    }
    PyErr_SetString(PyExc_TypeError,
                    "metadata residue_codes must be a string for pair-list");
    return false;
  }

  BufferView<const float, 2> buffer_;
  std::vector<char> residue_codes_;
  std::vector<universal::ResidueMetadataView> residues_;
  std::string source_id_;
  std::string source_filename_;
};

PyObject* string_view_to_python(std::string_view text) {
  return PythonCaster<std::string_view>::to_python(text);
}

PyObject* string_view_list_to_python(
    universal::Span<const std::string_view> values) {
  return span_to_pylist(values, [](std::string_view value) {
    return string_view_to_python(value);
  });
}

PyObject* availability_to_python(const api::BackendAvailability& availability) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_bool("compiled", availability.compiled) ||
      !dict.set_bool("runtime_available", availability.runtime_available) ||
      !dict.set_new_ref("reason",
                        string_view_to_python(availability.reason))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* gpu_availability_to_python(
    const api::GpuBackendAvailability& capability) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("compiled",
                        PythonCaster<bool>::to_python(
                            capability.availability.compiled)) ||
      !dict.set_new_ref(
          "runtime_available",
          PythonCaster<bool>::to_python(
              capability.availability.runtime_available)) ||
      !dict.set_new_ref("reason",
                        string_view_to_python(
                            capability.availability.reason)) ||
      !dict.set_new_ref("devices",
                        string_view_list_to_python(capability.devices))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* cpu_capabilities_to_python(
    const api::CpuBackendCapabilities& capabilities) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("scalar",
                        availability_to_python(capabilities.scalar)) ||
      !dict.set_new_ref("sse4", availability_to_python(capabilities.sse4)) ||
      !dict.set_new_ref("avx2", availability_to_python(capabilities.avx2)) ||
      !dict.set_new_ref("avx512",
                        availability_to_python(capabilities.avx512)) ||
      !dict.set_new_ref("neon", availability_to_python(capabilities.neon)) ||
      !dict.set_new_ref("sve", availability_to_python(capabilities.sve))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* gpu_capabilities_to_python(
    const api::GpuBackendCapabilities& capabilities) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("cuda",
                        gpu_availability_to_python(capabilities.cuda)) ||
      !dict.set_new_ref("hip",
                        gpu_availability_to_python(capabilities.hip)) ||
      !dict.set_new_ref("metal",
                        gpu_availability_to_python(capabilities.metal)) ||
      !dict.set_new_ref("vulkan",
                        gpu_availability_to_python(capabilities.vulkan)) ||
      !dict.set_new_ref("opencl",
                        gpu_availability_to_python(capabilities.opencl))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* pipeline_capabilities_to_python(
    const api::PipelineCapabilities& capabilities) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_bool("pairwise_alignment", capabilities.pairwise_alignment) ||
      !dict.set_bool("symmetric_all_vs_all",
                     capabilities.symmetric_all_vs_all) ||
      !dict.set_bool("structure_inputs", capabilities.structure_inputs) ||
      !dict.set_bool("embedding_inputs", capabilities.embedding_inputs)) {
    return nullptr;
  }
  return dict.release();
}

const char* package_kind_name(universal::PackageKind package_kind) noexcept {
  switch (package_kind) {
    case universal::PackageKind::RegisteredArchitecture:
      return "registered_architecture";
    case universal::PackageKind::GraphIr:
      return "graph_ir";
    case universal::PackageKind::SubstitutionMatrix:
      return "substitution_matrix";
    case universal::PackageKind::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* execution_mode_name(universal::PackageExecutionMode mode) noexcept {
  switch (mode) {
    case universal::PackageExecutionMode::RegisteredArchitecture:
      return "registered_architecture";
    case universal::PackageExecutionMode::GraphIr:
      return "graph_ir";
    case universal::PackageExecutionMode::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* score_method_name(universal::ScoreMethod method) noexcept {
  switch (method) {
    case universal::ScoreMethod::RawDotV1:
      return "raw_dot_v1";
    case universal::ScoreMethod::CosineV1:
      return "cosine_v1";
    case universal::ScoreMethod::SubstitutionLookupV1:
      return "substitution_lookup_v1";
    case universal::ScoreMethod::DirectScoreMatrixV1:
      return "direct_score_matrix_v1";
    case universal::ScoreMethod::LearnedPairScorerV1:
      return "learned_pair_scorer_v1";
  }
  return "unknown";
}

const char* score_normalization_name(
    universal::ScoreNormalization normalization) noexcept {
  switch (normalization) {
    case universal::ScoreNormalization::None:
      return "none";
    case universal::ScoreNormalization::L2:
      return "l2";
    case universal::ScoreNormalization::CalibratedLogOdds:
      return "calibrated_log_odds";
    case universal::ScoreNormalization::PackageSpecific:
      return "package_specific";
  }
  return "unknown";
}

const char* score_scale_family_name(
    universal::ScoreScaleFamily family) noexcept {
  switch (family) {
    case universal::ScoreScaleFamily::RawDot:
      return "raw_dot";
    case universal::ScoreScaleFamily::CosineUnitless:
      return "cosine_unitless";
    case universal::ScoreScaleFamily::LogOdds:
      return "log_odds";
    case universal::ScoreScaleFamily::LearnedLogit:
      return "learned_logit";
    case universal::ScoreScaleFamily::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* gap_model_name(universal::GapModel model) noexcept {
  switch (model) {
    case universal::GapModel::Affine:
      return "affine";
  }
  return "unknown";
}

const char* gap_convention_name(
    universal::GapConvention convention) noexcept {
  switch (convention) {
    case universal::GapConvention::GapOpenPlusKMinusOneGapExtension:
      return "gap_open_plus_k_minus_1_gap_extension";
  }
  return "unknown";
}

const char* alignment_algorithm_name(
    universal::AlignmentAlgorithmId algorithm) noexcept {
  switch (algorithm) {
    case universal::AlignmentAlgorithmId::HardLocalAffineSwV1:
      return "hard_local_affine_sw_v1";
    case universal::AlignmentAlgorithmId::GlobalAffineSwV1:
      return "global_affine_sw_v1";
    case universal::AlignmentAlgorithmId::SemiglobalAffineSwV1:
      return "semiglobal_affine_sw_v1";
    case universal::AlignmentAlgorithmId::SoftSwV1:
      return "soft_sw_v1";
  }
  return "unknown";
}

bool visible_model_record(
    const weights::PackageRegistryRecord& record) noexcept {
  return record.descriptor != nullptr &&
         record.descriptor->identity.package_kind ==
             universal::PackageKind::RegisteredArchitecture;
}

const weights::WeightManifestView& manifest_for_package(
    std::string_view package_id) noexcept {
  if (package_id == weights::kDefaultEsm2_8mModelName) {
    return weights::default_esm2_8m_manifest();
  }
  return weights::default_mpnn_d64_manifest();
}

PyObject* package_availability_to_python(
    const weights::PackageRegistryRecord& record) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_bool("compiled", record.compiled) ||
      !dict.set_bool("runtime_available", record.runtime_available) ||
      !dict.set_new_ref("reason", string_view_to_python(record.reason))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* score_semantics_to_python(
    const universal::ScoreSemantics& semantics) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("normalization",
                        PyUnicode_FromString(score_normalization_name(
                            semantics.normalization))) ||
      !dict.set_new_ref("scale_family",
                        PyUnicode_FromString(
                            score_scale_family_name(semantics.scale_family))) ||
      !dict.set_bool("higher_is_better", semantics.higher_is_better) ||
      !dict.set_bool("local_affine_additive",
                     semantics.local_affine_additive)) {
    return nullptr;
  }
  return dict.release();
}

PyObject* scoring_to_python(const universal::ScoringDescriptor& scoring,
                            const weights::WeightManifestView& manifest) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("method",
                        PyUnicode_FromString(score_method_name(
                            scoring.method))) ||
      !dict.set_new_ref("similarity",
                        string_view_to_python(manifest.similarity)) ||
      !dict.set_new_ref("semantics",
                        score_semantics_to_python(scoring.semantics))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* gaps_to_python(const universal::AffineGapDefaults& gaps) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("family", string_view_to_python(gaps.family)) ||
      !dict.set_new_ref("model",
                        PyUnicode_FromString(gap_model_name(gaps.model))) ||
      !dict.set_double("gap_open", gaps.gap_open) ||
      !dict.set_double("gap_extension", gaps.gap_extension) ||
      !dict.set_new_ref("convention",
                        PyUnicode_FromString(gap_convention_name(
                            gaps.convention))) ||
      !dict.set_new_ref(
          "calibrated_for_score_method",
          PyUnicode_FromString(score_method_name(
              gaps.calibrated_for_score_method)))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* checksums_to_python(const weights::WeightManifestView& manifest) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("algorithm",
                        string_view_to_python(
                            manifest.checksum_algorithm)) ||
      !dict.set_new_ref("package", string_view_to_python(manifest.checksum)) ||
      !dict.set_new_ref("source_checkpoint",
                        string_view_to_python(
                            manifest.source_checkpoint_checksum))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* provenance_to_python(const weights::WeightManifestView& manifest) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("source_checkpoint",
                        string_view_to_python(manifest.source_checkpoint)) ||
      !dict.set_new_ref("generation_tool",
                        string_view_to_python(manifest.generation_tool)) ||
      !dict.set_new_ref("generation_tool_version",
                        string_view_to_python(
                            manifest.generation_tool_version)) ||
      !dict.set_new_ref("generation_date",
                        string_view_to_python(manifest.generation_date)) ||
      !dict.set_new_ref("validation_status",
                        string_view_to_python(manifest.validation_status)) ||
      !dict.set_new_ref("status",
                        string_view_to_python(manifest.provenance_status))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* model_record_to_python(
    const weights::PackageRegistryRecord& record,
    const weights::WeightManifestView& manifest) {
  const universal::PackageDescriptor& descriptor = *record.descriptor;
  const std::size_t hidden_dimension =
      descriptor.compatibility_views.weights.view == nullptr
          ? manifest.hidden_dimension
          : descriptor.compatibility_views.weights.view->metadata
                .hidden_dimension;

  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref(
          "package_id",
          string_view_to_python(descriptor.identity.package_id)) ||
      !dict.set_new_ref("aliases",
                        string_view_list_to_python(
                            descriptor.identity.aliases)) ||
      !dict.set_new_ref(
          "family",
          string_view_to_python(descriptor.identity.package_family)) ||
      !dict.set_new_ref(
          "version",
          string_view_to_python(descriptor.identity.package_version)) ||
      !dict.set_new_ref("package_kind",
                        PyUnicode_FromString(package_kind_name(
                            descriptor.identity.package_kind))) ||
      !dict.set_new_ref(
          "execution_mode",
          PyUnicode_FromString(execution_mode_name(
              descriptor.execution.mode))) ||
      !dict.set_new_ref(
          "architecture_id",
          string_view_to_python(descriptor.execution.architecture_id)) ||
      !dict.set_size("hidden_dimension", hidden_dimension) ||
      !dict.set_new_ref("scoring",
                        scoring_to_python(descriptor.scoring, manifest)) ||
      !dict.set_new_ref("gaps", gaps_to_python(descriptor.gaps)) ||
      !dict.set_new_ref("soft_gaps",
                        gaps_to_python(descriptor.soft_gaps)) ||
      !dict.set_new_ref(
          "alignment_algorithm",
          PyUnicode_FromString(alignment_algorithm_name(
              descriptor.alignment.algorithm))) ||
      !dict.set_new_ref("checksum", checksums_to_python(manifest)) ||
      !dict.set_new_ref("provenance", provenance_to_python(manifest)) ||
      !dict.set_new_ref("availability",
                        package_availability_to_python(record))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* py_version_info(PyObject*, PyObject*) {
  const api::VersionInfo info = api::version_info();
  const std::string label(info.version.label);
  PyObjectRef version_tuple(Py_BuildValue("(iiis)", info.version.major,
                                          info.version.minor,
                                          info.version.patch, label.c_str()));
  DictBuilder dict;
  if (!version_tuple || !dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("product_name",
                        string_view_to_python(info.product_name)) ||
      !dict.set_borrowed("version", version_tuple.get()) ||
      !dict.set_long("major", info.version.major) ||
      !dict.set_long("minor", info.version.minor) ||
      !dict.set_long("patch", info.version.patch) ||
      !dict.set_new_ref("label", PyUnicode_FromString(label.c_str()))) {
    return nullptr;
  }
  return dict.release();
}

const char* backend_name(universal::Backend backend) noexcept {
  switch (backend) {
    case universal::Backend::Auto:
      return "auto";
    case universal::Backend::Scalar:
      return "scalar";
    case universal::Backend::Sse4:
      return "sse4";
    case universal::Backend::Avx2:
      return "avx2";
    case universal::Backend::Avx512:
      return "avx512";
    case universal::Backend::Neon:
      return "neon";
    case universal::Backend::Sve:
      return "sve";
    case universal::Backend::Cuda:
      return "cuda";
    case universal::Backend::Hip:
      return "hip";
    case universal::Backend::Metal:
      return "metal";
    case universal::Backend::Vulkan:
      return "vulkan";
    case universal::Backend::OpenCl:
      return "opencl";
  }
  return "unknown";
}

PyObject* py_backend_capabilities(PyObject*, PyObject*) {
  const api::BackendCapabilities capabilities = api::backend_capabilities();
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("cpu",
                        cpu_capabilities_to_python(capabilities.cpu)) ||
      !dict.set_new_ref("gpu",
                        gpu_capabilities_to_python(capabilities.gpu)) ||
      !dict.set_new_ref("pipeline",
                        pipeline_capabilities_to_python(
                            capabilities.pipeline)) ||
      !dict.set_new_ref(
          "default_backend",
          PyUnicode_FromString(backend_name(capabilities.default_backend)))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* py_compiled_models_info(PyObject*, PyObject*) {
  const universal::Span<const weights::PackageRegistryRecord> records =
      weights::compiled_packages();

  std::size_t visible_count = 0;
  for (std::size_t index = 0; index < records.size; ++index) {
    if (visible_model_record(records.data[index])) {
      ++visible_count;
    }
  }

  ListBuilder packages(visible_count);
  if (!packages) {
    return nullptr;
  }
  std::size_t package_index = 0;
  for (std::size_t index = 0; index < records.size; ++index) {
    const weights::PackageRegistryRecord& record = records.data[index];
    if (!visible_model_record(record)) {
      continue;
    }
    const weights::WeightManifestView& manifest =
        manifest_for_package(record.descriptor->identity.package_id);
    if (!packages.set_new_ref(package_index,
                              model_record_to_python(record, manifest))) {
      return nullptr;
    }
    ++package_index;
  }

  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_size("count", visible_count) ||
      !dict.set_new_ref("packages", packages.release())) {
    return nullptr;
  }
  return dict.release();
}

PyObject* py_default_weights_info(PyObject*, PyObject*) {
  const weights::WeightManifestView& manifest =
      weights::default_mpnn_d64_manifest();
  const auto handle = weights::default_mpnn_d64();
  if (raise_if_error(handle.status) < 0) {
    return nullptr;
  }

  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("model_name",
                        string_view_to_python(manifest.model_name)) ||
      !dict.set_new_ref("model_family",
                        string_view_to_python(manifest.model_family)) ||
      !dict.set_size("hidden_dimension", manifest.hidden_dimension) ||
      !dict.set_size("neighbor_count", manifest.neighbor_count) ||
      !dict.set_size("rbf_count", manifest.rbf_count) ||
      !dict.set_size("layer_count", manifest.layer_count) ||
      !dict.set_double("gap_open", manifest.gap_open) ||
      !dict.set_double("gap_extension", manifest.gap_extension) ||
      !dict.set_double("soft_gap_open", manifest.soft_gap_open) ||
      !dict.set_double("soft_gap_extension",
                       manifest.soft_gap_extension) ||
      !dict.set_new_ref("checksum",
                        string_view_to_python(manifest.checksum)) ||
      !dict.set_new_ref("similarity",
                        string_view_to_python(manifest.similarity))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* py_pairwise_from_embeddings(PyObject*,
                                      PyObject* args,
                                      PyObject* kwargs) {
  PyObject* query = nullptr;
  PyObject* target = nullptr;
  PyObject* query_metadata = Py_None;
  PyObject* target_metadata = Py_None;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "pairwise_from_embeddings");
  if (!parser.required_object("query_embeddings", query) ||
      !parser.required_object("target_embeddings", target) ||
      !parser.optional_object("query_metadata", query_metadata, Py_None) ||
      !parser.optional_object("target_metadata", target_metadata, Py_None) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parse_pairwise_mode_kwargs(parser, mode, temperature) ||
      !parser.finish()) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine = make_package_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  EmbeddingInput query_input;
  EmbeddingInput target_input;
  if (!query_input.acquire(query, query_metadata, "query embeddings") ||
      !target_input.acquire(target, target_metadata, "target embeddings")) {
    return nullptr;
  }

  api::PairwiseEmbeddingRequest request{};
  request.query = query_input.view();
  request.target = target_input.view();
  request.alignment.gap_open = static_cast<float>(gap_open);
  request.alignment.gap_extension = static_cast<float>(gap_extension);
  request.mode = mode;
  request.temperature = static_cast<float>(temperature);

  const auto result = engine.value.pairwise(request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pairwise_result_to_python(result.value);
}

PyObject* py_load_structure_metadata(PyObject*,
                                     PyObject* args,
                                     PyObject* kwargs) {
  PyObject* path_arg = nullptr;
  StructureOptionFields fields;
  TypedArgParser parser(args, kwargs, "load_structure_metadata");
  if (!parser.required_object("path", path_arg) ||
      !parse_structure_options(parser, fields) || !parser.finish()) {
    return nullptr;
  }

  io::StructureLoadOptions options{};
  if (apply_structure_options(fields, options) < 0) {
    return nullptr;
  }

  std::string path;
  if (!path_from_python(path_arg, path)) {
    return nullptr;
  }
  auto loaded = io::load_structure_from_file(path, options);
  if (raise_if_error(loaded.status) < 0) {
    return nullptr;
  }

  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_size("residue_count", loaded.value.residue_count()) ||
      !dict.set_new_ref("input_id",
                        string_view_to_python(loaded.value.input_id())) ||
      !dict.set_new_ref(
          "source_filename",
          string_view_to_python(loaded.value.source_filename())) ||
      !dict.set_new_ref(
          "selected_chain_id",
          string_view_to_python(loaded.value.selected_chain_id())) ||
      !dict.set_new_ref(
          "selected_model_id",
          string_view_to_python(loaded.value.selected_model_id())) ||
      !dict.set_long("selected_model_index",
                     loaded.value.selected_model_index())) {
    return nullptr;
  }
  return dict.release();
}

PyObject* py_encode_from_structure(PyObject*,
                                   PyObject* args,
                                   PyObject* kwargs) {
  PyObject* structure_arg = nullptr;
  PyObject* package = Py_None;
  StructureOptionFields fields;
  TypedArgParser parser(args, kwargs, "encode_from_structure");
  if (!parser.required_object("structure", structure_arg) ||
      !parser.optional_object("package", package, Py_None) ||
      !parse_structure_options(parser, fields) || !parser.finish()) {
    return nullptr;
  }

  io::StructureLoadOptions load_options{};
  if (apply_structure_options(fields, load_options) < 0) {
    return nullptr;
  }
  io::LoadedStructure loaded;
  if (!load_structure_arg(structure_arg, load_options, loaded)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine = make_package_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  api::EncodeStructureRequest request{};
  request.structure = loaded.view();
  const auto result = engine.value.encode(request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return encode_result_to_python(result.value);
}

PyObject* py_encode_from_coords(PyObject*, PyObject* args, PyObject* kwargs) {
  PyObject* coords_arg = nullptr;
  PyObject* package = Py_None;
  TypedArgParser parser(args, kwargs, "encode_from_coords");
  if (!parser.required_object("coords", coords_arg) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.finish()) {
    return nullptr;
  }

  io::LoadedStructure loaded;
  if (!load_structure_arg(coords_arg, io::StructureLoadOptions{}, loaded)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine = make_package_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  const universal::StructureView view = loaded.view();
  api::EncodeCoordsRequest request{};
  request.coords = coords_input_from_structure(view);
  const auto result = engine.value.encode(request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return encode_result_to_python(result.value);
}

PyObject* py_pairwise_from_structure(PyObject*,
                                     PyObject* args,
                                     PyObject* kwargs) {
  PyObject* query_arg = nullptr;
  PyObject* target_arg = nullptr;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  StructureOptionFields fields;
  TypedArgParser parser(args, kwargs, "pairwise_from_structure");
  if (!parser.required_object("query", query_arg) ||
      !parser.required_object("target", target_arg) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parse_structure_options(parser, fields) ||
      !parse_pairwise_mode_kwargs(parser, mode, temperature) ||
      !parser.finish()) {
    return nullptr;
  }

  io::StructureLoadOptions load_options{};
  if (apply_structure_options(fields, load_options) < 0) {
    return nullptr;
  }
  io::LoadedStructure query_loaded;
  io::LoadedStructure target_loaded;
  if (!load_structure_arg(query_arg, load_options, query_loaded) ||
      !load_structure_arg(target_arg, load_options, target_loaded)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine = make_package_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  api::PairwiseStructureRequest request{};
  request.query = query_loaded.view();
  request.target = target_loaded.view();
  request.alignment.gap_open = static_cast<float>(gap_open);
  request.alignment.gap_extension = static_cast<float>(gap_extension);
  request.mode = mode;
  request.temperature = static_cast<float>(temperature);
  const auto result = engine.value.pairwise(request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pairwise_result_to_python(result.value);
}

PyObject* py_pairwise_from_coords(PyObject*, PyObject* args, PyObject* kwargs) {
  PyObject* query_arg = nullptr;
  PyObject* target_arg = nullptr;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "pairwise_from_coords");
  if (!parser.required_object("query_coords", query_arg) ||
      !parser.required_object("target_coords", target_arg) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parse_pairwise_mode_kwargs(parser, mode, temperature) ||
      !parser.finish()) {
    return nullptr;
  }

  io::LoadedStructure query_loaded;
  io::LoadedStructure target_loaded;
  if (!load_structure_arg(query_arg, io::StructureLoadOptions{},
                          query_loaded) ||
      !load_structure_arg(target_arg, io::StructureLoadOptions{},
                          target_loaded)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine = make_package_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  const universal::StructureView query_view = query_loaded.view();
  const universal::StructureView target_view = target_loaded.view();
  api::PairwiseCoordsRequest request{};
  request.query = coords_input_from_structure(query_view);
  request.target = coords_input_from_structure(target_view);
  request.alignment.gap_open = static_cast<float>(gap_open);
  request.alignment.gap_extension = static_cast<float>(gap_extension);
  request.mode = mode;
  request.temperature = static_cast<float>(temperature);
  const auto result = engine.value.pairwise(request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pairwise_result_to_python(result.value);
}

PyObject* py_encode_from_sequence(PyObject*,
                                  PyObject* args,
                                  PyObject* kwargs) {
  PyObject* sequence_arg = nullptr;
  PyObject* package = Py_None;
  TypedArgParser parser(args, kwargs, "encode_from_sequence");
  if (!parser.required_object("sequence", sequence_arg) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.finish()) {
    return nullptr;
  }

  std::vector<std::int32_t> tokens;
  if (!tokenize_python_sequence(sequence_arg, "sequence", tokens)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine = make_package_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  api::EncodeSequenceRequest request{};
  request.token_ids = {tokens.data(), tokens.size()};
  const auto result = engine.value.encode(request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return encode_result_to_python(result.value);
}

PyObject* py_pairwise_from_sequence(PyObject*,
                                    PyObject* args,
                                    PyObject* kwargs) {
  PyObject* query_arg = nullptr;
  PyObject* target_arg = nullptr;
  PyObject* package = Py_None;
  // Default to the package-default sentinel (NaN) so the engine's sequence
  // route resolves the gap values from the resolved package descriptor's
  // calibrated parameters. Callers that supply `gap_open=` or
  // `gap_extension=` keyword arguments still get their explicit values
  // threaded through unchanged. fe2 traced bl5's soft-SW pilot collapse
  // to this binding silently injecting MPNN-64 defaults (-1.4 / -0.15)
  // into ESM2-8M sequence calls.
  const double kSentinel =
      static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_open = kSentinel;
  double gap_extension = kSentinel;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "pairwise_from_sequence");
  if (!parser.required_object("query", query_arg) ||
      !parser.required_object("target", target_arg) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional("gap_open", gap_open, kSentinel) ||
      !parser.optional("gap_extension", gap_extension, kSentinel) ||
      !parse_pairwise_mode_kwargs(parser, mode, temperature) ||
      !parser.finish()) {
    return nullptr;
  }

  std::vector<std::int32_t> query_tokens;
  std::vector<std::int32_t> target_tokens;
  if (!tokenize_python_sequence(query_arg, "query", query_tokens) ||
      !tokenize_python_sequence(target_arg, "target", target_tokens)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine = make_package_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  api::PairwiseSequenceRequest request{};
  request.query_token_ids = {query_tokens.data(), query_tokens.size()};
  request.target_token_ids = {target_tokens.data(), target_tokens.size()};
  request.alignment.gap_open = static_cast<float>(gap_open);
  request.alignment.gap_extension = static_cast<float>(gap_extension);
  request.mode = mode;
  request.temperature = static_cast<float>(temperature);

  const auto result = engine.value.pairwise(request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pairwise_result_to_python(result.value);
}

PyObject* py_pair_list_from_sequence(PyObject*,
                                     PyObject* args,
                                     PyObject* kwargs) {
  PyObject* pairs_arg = nullptr;
  PyObject* fasta_path_arg = nullptr;
  PyObject* package = Py_None;
  PyObject* threads_arg = nullptr;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  const double kSentinel =
      static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_open = kSentinel;
  double gap_extension = kSentinel;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "pair_list_from_sequence");
  if (!parser.required_object("pairs", pairs_arg) ||
      !parser.required_object("fasta_path", fasta_path_arg) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional("gap_open", gap_open, kSentinel) ||
      !parser.optional("gap_extension", gap_extension, kSentinel) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parser.optional_object("threads", threads_arg, nullptr) ||
      !parse_pair_list_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() ||
      !parse_pair_list_thread_kwargs(threads_arg, thread_count_arg,
                                     thread_count)) {
    return nullptr;
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  if (!parse_pair_list_pairs(pairs_arg, pairs)) {
    return nullptr;
  }

  std::string fasta_path;
  if (!path_from_python(fasta_path_arg, fasta_path)) {
    return nullptr;
  }
  FastaRecords fasta;
  universal::Status status = read_fasta_records(fasta_path, fasta);
  if (raise_if_error(status) < 0) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine =
      make_sequence_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  std::vector<api::SequenceEntry> entries;
  entries.reserve(fasta.tokens.size());
  for (std::size_t index = 0; index < fasta.tokens.size(); ++index) {
    api::SequenceEntry entry{};
    entry.name = fasta.names[index];
    entry.token_ids = {fasta.tokens[index].data(), fasta.tokens[index].size()};
    entries.push_back(entry);
  }

  api::PairListSequenceRequest request{};
  request.sequences = {entries.empty() ? nullptr : entries.data(),
                       entries.size()};
  request.pairs = pairs;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const auto result = collect_pair_list_without_gil(engine.value, request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pair_list_result_to_python(result.value);
}

PyObject* py_pair_list_from_structure(PyObject*,
                                      PyObject* args,
                                      PyObject* kwargs) {
  PyObject* pairs_arg = nullptr;
  PyObject* pdb_dir_arg = nullptr;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* threads_arg = nullptr;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  StructureOptionFields fields;
  TypedArgParser parser(args, kwargs, "pair_list_from_structure");
  if (!parser.required_object("pairs", pairs_arg) ||
      !parser.required_object("pdb_dir", pdb_dir_arg) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parse_structure_options(parser, fields) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parser.optional_object("threads", threads_arg, nullptr) ||
      !parse_pair_list_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() ||
      !parse_pair_list_thread_kwargs(threads_arg, thread_count_arg,
                                     thread_count)) {
    return nullptr;
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  if (!parse_pair_list_pairs(pairs_arg, pairs)) {
    return nullptr;
  }

  io::StructureLoadOptions load_options{};
  if (apply_structure_options(fields, load_options) < 0) {
    return nullptr;
  }
  std::vector<io::LoadedStructure> loaded;
  std::vector<universal::StructureView> views;
  if (!load_structures_from_directory(pdb_dir_arg, load_options, loaded,
                                      views)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  api::PairListStructureRequest request{};
  request.structures = {views.empty() ? nullptr : views.data(), views.size()};
  request.pairs = pairs;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const auto result = collect_pair_list_without_gil(engine.value, request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pair_list_result_to_python(result.value);
}

PyObject* py_pair_list_from_coords(PyObject*, PyObject* args, PyObject* kwargs) {
  PyObject* pairs_arg = nullptr;
  PyObject* inputs_arg = nullptr;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* threads_arg = nullptr;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "pair_list_from_coords");
  if (!parser.required_object("pairs", pairs_arg) ||
      !parser.required_object("coords", inputs_arg) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parser.optional_object("threads", threads_arg, nullptr) ||
      !parse_pair_list_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() ||
      !parse_pair_list_thread_kwargs(threads_arg, thread_count_arg,
                                     thread_count)) {
    return nullptr;
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  if (!parse_pair_list_pairs(pairs_arg, pairs)) {
    return nullptr;
  }

  std::vector<io::LoadedStructure> loaded;
  std::vector<universal::StructureView> views;
  if (!load_structure_sequence(inputs_arg, io::StructureLoadOptions{}, loaded,
                               views)) {
    return nullptr;
  }
  std::vector<api::CoordsInputView> coords;
  coords.reserve(views.size());
  for (const universal::StructureView& view : views) {
    coords.push_back(coords_input_from_structure(view));
  }

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  api::PairListCoordsRequest request{};
  request.coords = {coords.empty() ? nullptr : coords.data(), coords.size()};
  request.pairs = pairs;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const auto result = collect_pair_list_without_gil(engine.value, request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pair_list_result_to_python(result.value);
}

PyObject* py_pair_list_from_embeddings(PyObject*,
                                       PyObject* args,
                                       PyObject* kwargs) {
  PyObject* pairs_arg = nullptr;
  PyObject* embeddings_arg = nullptr;
  PyObject* metadata_arg = Py_None;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* threads_arg = nullptr;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "pair_list_from_embeddings");
  if (!parser.required_object("pairs", pairs_arg) ||
      !parser.required_object("embeddings", embeddings_arg) ||
      !parser.optional_object("metadata", metadata_arg, Py_None) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parser.optional_object("threads", threads_arg, nullptr) ||
      !parse_pair_list_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() ||
      !parse_pair_list_thread_kwargs(threads_arg, thread_count_arg,
                                     thread_count)) {
    return nullptr;
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  if (!parse_pair_list_pairs(pairs_arg, pairs)) {
    return nullptr;
  }

  PyObjectRef sequence(
      PySequence_Fast(embeddings_arg, "embeddings must be a sequence"));
  if (!sequence) {
    return nullptr;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());

  std::vector<std::unique_ptr<NamedEmbeddingInput>> inputs;
  std::vector<universal::EmbeddingView> views;
  inputs.reserve(static_cast<std::size_t>(count));
  views.reserve(static_cast<std::size_t>(count));

  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < count; ++index) {
    auto input = std::make_unique<NamedEmbeddingInput>();
    PyObjectRef metadata = metadata_for_index(metadata_arg, index);
    if (!metadata) {
      return nullptr;
    }
    if (!input->acquire(items[index], metadata.get(), "embedding input")) {
      return nullptr;
    }
    views.push_back(input->view());
    inputs.push_back(std::move(input));
  }

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  api::PairListEmbeddingRequest request{};
  request.embeddings = {views.empty() ? nullptr : views.data(), views.size()};
  request.pairs = pairs;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const auto result = collect_pair_list_without_gil(engine.value, request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return pair_list_result_to_python(result.value);
}

PyMethodDef kMethods[] = {
    {"version_info", py_version_info, METH_NOARGS,
     "Return Hikoboshi version information."},
    {"backend_capabilities", py_backend_capabilities, METH_NOARGS,
     "Return Hikoboshi backend capability flags."},
    {"default_weights_info", py_default_weights_info, METH_NOARGS,
     "Return metadata for the compiled hikoboshi-mpnn-d64 weights."},
    {"compiled_models_info", py_compiled_models_info, METH_NOARGS,
     "Return compiled Hikoboshi package registry model information."},
    {"pairwise_from_embeddings",
     reinterpret_cast<PyCFunction>(py_pairwise_from_embeddings),
     METH_VARARGS | METH_KEYWORDS,
     "Run pairwise alignment from two float32 embedding arrays. Defaults to "
     "hard Smith-Waterman; pass mode='soft' or mode='both' to opt in to "
     "soft scoring."},
    {"load_structure_metadata",
     reinterpret_cast<PyCFunction>(py_load_structure_metadata),
     METH_VARARGS | METH_KEYWORDS,
     "Load structure metadata through the Hikoboshi IO layer."},
    {"encode_from_structure",
     reinterpret_cast<PyCFunction>(py_encode_from_structure),
     METH_VARARGS | METH_KEYWORDS,
     "Encode a structure path through the public Hikoboshi API."},
    {"encode_from_coords",
     reinterpret_cast<PyCFunction>(py_encode_from_coords),
     METH_VARARGS | METH_KEYWORDS,
     "Encode path-backed coordinates through the public Hikoboshi API."},
    {"pairwise_from_structure",
     reinterpret_cast<PyCFunction>(py_pairwise_from_structure),
     METH_VARARGS | METH_KEYWORDS,
     "Run pairwise alignment over structure paths. Defaults to hard "
     "Smith-Waterman; pass mode='soft' or mode='both' to opt in to soft "
     "scoring."},
    {"pairwise_from_coords",
     reinterpret_cast<PyCFunction>(py_pairwise_from_coords),
     METH_VARARGS | METH_KEYWORDS,
     "Run pairwise alignment over path-backed coordinates. Defaults to hard "
     "Smith-Waterman; pass mode='soft' or mode='both' to opt in to soft "
     "scoring."},
    {"encode_from_sequence",
     reinterpret_cast<PyCFunction>(py_encode_from_sequence),
     METH_VARARGS | METH_KEYWORDS,
     "Encode an AA sequence string through a sequence-input package such as "
     "hikoboshi-esm2-8m. Returns per-residue embeddings."},
    {"pairwise_from_sequence",
     reinterpret_cast<PyCFunction>(py_pairwise_from_sequence),
     METH_VARARGS | METH_KEYWORDS,
     "Run pairwise alignment over two AA sequence strings via a sequence-"
     "input package such as hikoboshi-esm2-8m. Defaults to hard "
     "Smith-Waterman; pass mode='soft' or mode='both' to opt in to soft "
     "scoring."},
    {"pair_list_from_sequence",
     reinterpret_cast<PyCFunction>(py_pair_list_from_sequence),
     METH_VARARGS | METH_KEYWORDS,
     "Run pair-list alignment over a named FASTA source. Returns one "
     "pairwise-result payload per input pair in input order."},
    {"pair_list_from_structure",
     reinterpret_cast<PyCFunction>(py_pair_list_from_structure),
     METH_VARARGS | METH_KEYWORDS,
     "Run pair-list alignment over a directory of PDB/mmCIF structures. "
     "Returns one pairwise-result payload per input pair in input order."},
    {"pair_list_from_coords",
     reinterpret_cast<PyCFunction>(py_pair_list_from_coords),
     METH_VARARGS | METH_KEYWORDS,
     "Run pair-list alignment over path-backed coordinate inputs. Returns "
     "one pairwise-result payload per input pair in input order."},
    {"pair_list_from_embeddings",
     reinterpret_cast<PyCFunction>(py_pair_list_from_embeddings),
     METH_VARARGS | METH_KEYWORDS,
     "Run pair-list alignment over named embedding arrays. Metadata entries "
     "must provide input_id or source_id for pair lookup."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

int bind_engine(PyObject* module) {
  for (PyMethodDef* method = kMethods; method->ml_name != nullptr; ++method) {
    PyObjectRef function(PyCFunction_NewEx(method, nullptr, nullptr));
    if (!function) {
      return -1;
    }
    if (PyModule_AddObject(module, method->ml_name, function.release()) < 0) {
      return -1;
    }
  }
  return 0;
}

}  // namespace hikoboshi::bindings
