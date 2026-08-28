#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/bindings/args.hpp>
#include <hikoboshi/bindings/engine_helpers.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hikoboshi::bindings {

PyObject* all_vs_all_result_to_python(const api::AllVsAllResult& result);

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

bool parse_all_vs_all_mode_kwargs(TypedArgParser& parser,
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
        "all_vs_all mode must be 'hard', 'soft', or 'both'"));
    return false;
  }
  return true;
}

bool alignment_mode_runs_soft(api::AlignmentMode mode) noexcept {
  return mode == api::AlignmentMode::Soft || mode == api::AlignmentMode::Both;
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

bool load_structures(PyObject* inputs_arg,
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

bool parse_include_self(PyObject* object, bool& include_self) {
  const int truth = PyObject_IsTrue(object);
  if (truth < 0) {
    return false;
  }
  include_self = truth != 0;
  return true;
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

template <typename Request>
universal::Result<api::AllVsAllResult> collect_all_vs_all_without_gil(
    const api::Engine& engine,
    const Request& request) {
  ScopedGilRelease gil_release;
  // p46: the record-returning Python entry points still build an
  // AllVsAllResult so existing callers that consume per-record metrics keep
  // working. The new `to_tsv` Python entry point uses
  // `api::stream_all_vs_all` and never reaches this helper. The deprecation
  // marker is intentionally suppressed at this single internal call site.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  return engine.collect_all_vs_all(request);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

struct StreamingTsvSummary {
  std::size_t pair_count = 0;
  double wall_time_seconds = 0.0;
};

template <typename Request>
universal::Status stream_all_vs_all_to_tsv_without_gil(
    const api::Engine& engine,
    const Request& request,
    const std::string& output_path,
    StreamingTsvSummary& summary) {
  ScopedGilRelease gil_release;
  std::ofstream file(output_path, std::ios::binary);
  if (!file) {
    return {universal::StatusCode::Unavailable,
            "all-vs-all summary path is not writable"};
  }
  api::TsvStreamingAllVsAllSink::Callbacks callbacks{};
  callbacks.include_dual_score_schema =
      alignment_mode_runs_soft(request.options.mode);
  api::TsvStreamingAllVsAllSink sink(file, callbacks);

  // Wall time is measured around stream_all_vs_all itself so the Python
  // caller receives an honest per-call duration even when the caller wraps
  // the binding with retries. CLOCK_MONOTONIC via steady_clock keeps the
  // measurement immune to system clock adjustments during the run.
  const auto start = std::chrono::steady_clock::now();
  universal::Status status = api::stream_all_vs_all(engine, request, sink);
  const auto end = std::chrono::steady_clock::now();
  if (!file.good()) {
    if (status.code == universal::StatusCode::Ok) {
      status = {universal::StatusCode::Unavailable,
                "all-vs-all summary write failed"};
    }
  }
  const std::chrono::duration<double> elapsed = end - start;
  summary.wall_time_seconds = elapsed.count();
  // The streaming sink emits exactly one row per `receive`. Counting via the
  // sink's emitted row counter would be marginally cheaper but the
  // public sink does not expose it; use the symmetric pair-count formula
  // applied to the request input span instead.
  summary.pair_count = 0;
  return status;
}

PyObject* python_summary_dict(const StreamingTsvSummary& summary,
                              const std::string& output_path) {
  PyObjectRef dict(PyDict_New());
  if (!dict) {
    return nullptr;
  }
  PyObjectRef pair_count(PyLong_FromUnsignedLongLong(
      static_cast<unsigned long long>(summary.pair_count)));
  if (!pair_count ||
      PyDict_SetItemString(dict.get(), "pair_count", pair_count.get()) < 0) {
    return nullptr;
  }
  PyObjectRef wall(PyFloat_FromDouble(summary.wall_time_seconds));
  if (!wall ||
      PyDict_SetItemString(dict.get(), "wall_time_seconds", wall.get()) < 0) {
    return nullptr;
  }
  PyObjectRef path(
      PyUnicode_FromStringAndSize(output_path.c_str(), output_path.size()));
  if (!path || PyDict_SetItemString(dict.get(), "output_path", path.get()) < 0) {
    return nullptr;
  }
  return dict.release();
}

bool parse_output_path_arg(PyObject* object, std::string& path) {
  if (object == nullptr) {
    PyErr_SetString(PyExc_TypeError, "output_path is required");
    return false;
  }
  if (!path_from_python(object, path)) {
    return false;
  }
  if (path.empty()) {
    PyErr_SetString(PyExc_ValueError, "output_path must not be empty");
    return false;
  }
  return true;
}

std::size_t symmetric_pair_count(std::size_t input_count,
                                 bool include_self) noexcept {
  if (input_count == 0U || (!include_self && input_count == 1U)) {
    return 0U;
  }
  const std::size_t other = include_self ? input_count + 1U : input_count - 1U;
  if (input_count % 2U == 0U) {
    return (input_count / 2U) * other;
  }
  return input_count * (other / 2U);
}

PyObject* py_all_vs_all_from_embeddings(PyObject*,
                                        PyObject* args,
                                        PyObject* kwargs) {
  PyObject* embeddings_arg = nullptr;
  PyObject* metadata_arg = Py_None;
  PyObject* include_self_arg = Py_False;
  bool include_self = false;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "all_vs_all_from_embeddings");
  if (!parser.required_object("embeddings", embeddings_arg) ||
      !parser.optional_object("metadata", metadata_arg, Py_None) ||
      !parser.optional_object("include_self", include_self_arg, Py_False) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parse_all_vs_all_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() || !parse_include_self(include_self_arg, include_self) ||
      !parse_thread_count_arg(thread_count_arg, thread_count)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  PyObjectRef sequence(
      PySequence_Fast(embeddings_arg, "embeddings must be a sequence"));
  if (!sequence) {
    return nullptr;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());

  std::vector<std::unique_ptr<EmbeddingInput>> inputs;
  std::vector<universal::EmbeddingView> views;
  inputs.reserve(static_cast<std::size_t>(count));
  views.reserve(static_cast<std::size_t>(count));

  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < count; ++index) {
    auto input = std::make_unique<EmbeddingInput>();
    PyObjectRef metadata = metadata_for_index(metadata_arg, index);
    if (!metadata) {
      return nullptr;
    }
    if (!input->acquire(items[index], metadata.get(), "embedding input",
                        false)) {
      return nullptr;
    }
    views.push_back(input->view());
    inputs.push_back(std::move(input));
  }

  api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.empty() ? nullptr : views.data(), views.size()};
  request.options.include_self = include_self;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const auto result = collect_all_vs_all_without_gil(engine.value, request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return all_vs_all_result_to_python(result.value);
}

PyObject* py_all_vs_all_from_structure(PyObject*,
                                       PyObject* args,
                                       PyObject* kwargs) {
  PyObject* inputs_arg = nullptr;
  PyObject* include_self_arg = Py_False;
  bool include_self = false;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  StructureOptionFields fields;
  TypedArgParser parser(args, kwargs, "all_vs_all_from_structure");
  if (!parser.required_object("inputs", inputs_arg) ||
      !parser.optional_object("include_self", include_self_arg, Py_False) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parse_structure_options(parser, fields) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parse_all_vs_all_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() || !parse_include_self(include_self_arg, include_self) ||
      !parse_thread_count_arg(thread_count_arg, thread_count)) {
    return nullptr;
  }

  io::StructureLoadOptions load_options{};
  if (apply_structure_options(fields, load_options) < 0) {
    return nullptr;
  }
  std::vector<io::LoadedStructure> loaded;
  std::vector<universal::StructureView> views;
  if (!load_structures(inputs_arg, load_options, loaded, views)) {
    return nullptr;
  }

  api::AllVsAllStructureRequest request{};
  request.structures = {views.empty() ? nullptr : views.data(), views.size()};
  request.options.include_self = include_self;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }
  const auto result = collect_all_vs_all_without_gil(engine.value, request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return all_vs_all_result_to_python(result.value);
}

PyObject* py_all_vs_all_from_coords(PyObject*,
                                    PyObject* args,
                                    PyObject* kwargs) {
  PyObject* inputs_arg = nullptr;
  PyObject* include_self_arg = Py_False;
  bool include_self = false;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "all_vs_all_from_coords");
  if (!parser.required_object("coords", inputs_arg) ||
      !parser.optional_object("include_self", include_self_arg, Py_False) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parse_all_vs_all_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() || !parse_include_self(include_self_arg, include_self) ||
      !parse_thread_count_arg(thread_count_arg, thread_count)) {
    return nullptr;
  }

  std::vector<io::LoadedStructure> loaded;
  std::vector<universal::StructureView> views;
  if (!load_structures(inputs_arg, io::StructureLoadOptions{}, loaded, views)) {
    return nullptr;
  }

  std::vector<api::CoordsInputView> coords;
  coords.reserve(views.size());
  for (const universal::StructureView& view : views) {
    coords.push_back(coords_input_from_structure(view));
  }

  api::AllVsAllCoordsRequest request{};
  request.coords = {coords.empty() ? nullptr : coords.data(), coords.size()};
  request.options.include_self = include_self;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }
  const auto result = collect_all_vs_all_without_gil(engine.value, request);
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }
  return all_vs_all_result_to_python(result.value);
}

PyObject* py_all_vs_all_to_tsv_from_embeddings(PyObject*,
                                               PyObject* args,
                                               PyObject* kwargs) {
  PyObject* embeddings_arg = nullptr;
  PyObject* output_path_arg = nullptr;
  PyObject* metadata_arg = Py_None;
  PyObject* include_self_arg = Py_False;
  bool include_self = false;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "all_vs_all_to_tsv_from_embeddings");
  if (!parser.required_object("embeddings", embeddings_arg) ||
      !parser.required_object("output_path", output_path_arg) ||
      !parser.optional_object("metadata", metadata_arg, Py_None) ||
      !parser.optional_object("include_self", include_self_arg, Py_False) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parse_all_vs_all_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() || !parse_include_self(include_self_arg, include_self) ||
      !parse_thread_count_arg(thread_count_arg, thread_count)) {
    return nullptr;
  }
  std::string output_path;
  if (!parse_output_path_arg(output_path_arg, output_path)) {
    return nullptr;
  }

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  PyObjectRef sequence(
      PySequence_Fast(embeddings_arg, "embeddings must be a sequence"));
  if (!sequence) {
    return nullptr;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());

  std::vector<std::unique_ptr<EmbeddingInput>> inputs;
  std::vector<universal::EmbeddingView> views;
  inputs.reserve(static_cast<std::size_t>(count));
  views.reserve(static_cast<std::size_t>(count));

  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < count; ++index) {
    auto input = std::make_unique<EmbeddingInput>();
    PyObjectRef metadata = metadata_for_index(metadata_arg, index);
    if (!metadata) {
      return nullptr;
    }
    if (!input->acquire(items[index], metadata.get(), "embedding input",
                        false)) {
      return nullptr;
    }
    views.push_back(input->view());
    inputs.push_back(std::move(input));
  }

  api::AllVsAllEmbeddingRequest request{};
  request.embeddings = {views.empty() ? nullptr : views.data(), views.size()};
  request.options.include_self = include_self;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  StreamingTsvSummary summary;
  const universal::Status status = stream_all_vs_all_to_tsv_without_gil(
      engine.value, request, output_path, summary);
  summary.pair_count = symmetric_pair_count(views.size(), include_self);
  if (raise_if_error(status) < 0) {
    return nullptr;
  }
  return python_summary_dict(summary, output_path);
}

PyObject* py_all_vs_all_to_tsv_from_structure(PyObject*,
                                              PyObject* args,
                                              PyObject* kwargs) {
  PyObject* inputs_arg = nullptr;
  PyObject* output_path_arg = nullptr;
  PyObject* include_self_arg = Py_False;
  bool include_self = false;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  StructureOptionFields fields;
  TypedArgParser parser(args, kwargs, "all_vs_all_to_tsv_from_structure");
  if (!parser.required_object("inputs", inputs_arg) ||
      !parser.required_object("output_path", output_path_arg) ||
      !parser.optional_object("include_self", include_self_arg, Py_False) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parse_structure_options(parser, fields) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parse_all_vs_all_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() || !parse_include_self(include_self_arg, include_self) ||
      !parse_thread_count_arg(thread_count_arg, thread_count)) {
    return nullptr;
  }
  std::string output_path;
  if (!parse_output_path_arg(output_path_arg, output_path)) {
    return nullptr;
  }

  io::StructureLoadOptions load_options{};
  if (apply_structure_options(fields, load_options) < 0) {
    return nullptr;
  }
  std::vector<io::LoadedStructure> loaded;
  std::vector<universal::StructureView> views;
  if (!load_structures(inputs_arg, load_options, loaded, views)) {
    return nullptr;
  }

  api::AllVsAllStructureRequest request{};
  request.structures = {views.empty() ? nullptr : views.data(), views.size()};
  request.options.include_self = include_self;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  StreamingTsvSummary summary;
  const universal::Status status = stream_all_vs_all_to_tsv_without_gil(
      engine.value, request, output_path, summary);
  summary.pair_count = symmetric_pair_count(views.size(), include_self);
  if (raise_if_error(status) < 0) {
    return nullptr;
  }
  return python_summary_dict(summary, output_path);
}

PyObject* py_all_vs_all_to_tsv_from_coords(PyObject*,
                                           PyObject* args,
                                           PyObject* kwargs) {
  PyObject* inputs_arg = nullptr;
  PyObject* output_path_arg = nullptr;
  PyObject* include_self_arg = Py_False;
  bool include_self = false;
  double gap_open = static_cast<double>(api::kPackageDefaultGapSentinel);
  double gap_extension = static_cast<double>(api::kPackageDefaultGapSentinel);
  PyObject* package = Py_None;
  PyObject* thread_count_arg = nullptr;
  std::uint32_t thread_count = 0;
  api::AlignmentMode mode = api::AlignmentMode::Hard;
  double temperature = api::kDefaultSoftTemperature;
  TypedArgParser parser(args, kwargs, "all_vs_all_to_tsv_from_coords");
  if (!parser.required_object("coords", inputs_arg) ||
      !parser.required_object("output_path", output_path_arg) ||
      !parser.optional_object("include_self", include_self_arg, Py_False) ||
      !parser.optional("gap_open", gap_open,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional("gap_extension", gap_extension,
                       static_cast<double>(api::kPackageDefaultGapSentinel)) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional_object("thread_count", thread_count_arg, nullptr) ||
      !parse_all_vs_all_mode_kwargs(parser, mode, temperature) ||
      !parser.finish() || !parse_include_self(include_self_arg, include_self) ||
      !parse_thread_count_arg(thread_count_arg, thread_count)) {
    return nullptr;
  }
  std::string output_path;
  if (!parse_output_path_arg(output_path_arg, output_path)) {
    return nullptr;
  }

  std::vector<io::LoadedStructure> loaded;
  std::vector<universal::StructureView> views;
  if (!load_structures(inputs_arg, io::StructureLoadOptions{}, loaded, views)) {
    return nullptr;
  }

  std::vector<api::CoordsInputView> coords;
  coords.reserve(views.size());
  for (const universal::StructureView& view : views) {
    coords.push_back(coords_input_from_structure(view));
  }

  api::AllVsAllCoordsRequest request{};
  request.coords = {coords.empty() ? nullptr : coords.data(), coords.size()};
  request.options.include_self = include_self;
  request.options.alignment.gap_open = static_cast<float>(gap_open);
  request.options.alignment.gap_extension = static_cast<float>(gap_extension);
  request.options.mode = mode;
  request.options.temperature = static_cast<float>(temperature);

  const universal::Result<api::Engine> engine =
      make_package_engine(package, thread_count);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  StreamingTsvSummary summary;
  const universal::Status status = stream_all_vs_all_to_tsv_without_gil(
      engine.value, request, output_path, summary);
  summary.pair_count = symmetric_pair_count(views.size(), include_self);
  if (raise_if_error(status) < 0) {
    return nullptr;
  }
  return python_summary_dict(summary, output_path);
}

PyMethodDef kMethods[] = {
    {"all_vs_all_from_embeddings",
     reinterpret_cast<PyCFunction>(py_all_vs_all_from_embeddings),
     METH_VARARGS | METH_KEYWORDS,
     "Run symmetric all-vs-all alignment over embedding arrays; defaults to "
     "hard Smith-Waterman; pass mode='soft' or mode='both' (with "
     "temperature) to opt in to soft scoring. thread_count selects "
     "all-vs-all workers."},
    {"all_vs_all_from_structure",
     reinterpret_cast<PyCFunction>(py_all_vs_all_from_structure),
     METH_VARARGS | METH_KEYWORDS,
     "Run symmetric all-vs-all alignment over structure paths; defaults to "
     "hard Smith-Waterman; pass mode='soft' or mode='both' (with "
     "temperature) to opt in to soft scoring. thread_count selects "
     "all-vs-all workers."},
    {"all_vs_all_from_coords",
     reinterpret_cast<PyCFunction>(py_all_vs_all_from_coords),
     METH_VARARGS | METH_KEYWORDS,
     "Run symmetric all-vs-all alignment over path-backed coordinates; "
     "defaults to hard Smith-Waterman; pass mode='soft' or mode='both' "
     "(with temperature) to opt in to soft scoring. thread_count selects all-vs-all "
     "workers."},
    {"all_vs_all_to_tsv_from_embeddings",
     reinterpret_cast<PyCFunction>(py_all_vs_all_to_tsv_from_embeddings),
     METH_VARARGS | METH_KEYWORDS,
     "Stream symmetric all-vs-all alignment over embedding arrays directly "
     "to a TSV at output_path; never materializes the full record list. "
     "Defaults to hard Smith-Waterman; pass mode='soft' or mode='both' "
     "(with temperature) to opt in to soft scoring."},
    {"all_vs_all_to_tsv_from_structure",
     reinterpret_cast<PyCFunction>(py_all_vs_all_to_tsv_from_structure),
     METH_VARARGS | METH_KEYWORDS,
     "Stream symmetric all-vs-all alignment over structure paths directly "
     "to a TSV at output_path; never materializes the full record list. "
     "Defaults to hard Smith-Waterman; pass mode='soft' or mode='both' "
     "(with temperature) to opt in to soft scoring."},
    {"all_vs_all_to_tsv_from_coords",
     reinterpret_cast<PyCFunction>(py_all_vs_all_to_tsv_from_coords),
     METH_VARARGS | METH_KEYWORDS,
     "Stream symmetric all-vs-all alignment over path-backed coordinates "
     "directly to a TSV at output_path; never materializes the full record "
     "list. Defaults to hard Smith-Waterman; pass mode='soft' or "
     "mode='both' (with temperature) to opt in to soft scoring."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

int bind_all_vs_all(PyObject* module) {
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
