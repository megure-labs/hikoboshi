#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/bindings/args.hpp>
#include <hikoboshi/bindings/casters.hpp>
#include <hikoboshi/bindings/dict_builder.hpp>
#include <hikoboshi/bindings/engine_helpers.hpp>
#include <hikoboshi/bindings/list_builder.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>
#include <hikoboshi/io/design_fasta_writer.hpp>
#include <hikoboshi/io/structure_loader.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace hikoboshi::bindings {
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

universal::Result<api::Engine> make_inverse_fold_engine(
    PyObject* package_object) {
  const universal::Result<weights::PackageHandle> package =
      (package_object == nullptr || package_object == Py_None)
          ? weights::default_package(
                weights::kDefaultProteinMpnnV48Eps020ModelName)
          : package_from_python(package_object);
  if (!status_ok(package.status)) {
    return {package.status, api::Engine{}};
  }
  if (package.value.descriptor == nullptr) {
    return {failed_precondition(
                "default ProteinMPNN package descriptor is missing"),
            api::Engine{}};
  }

  api::EngineConfig config{};
  config.package = package.value;
  config.weights = package.value.descriptor->compatibility_views.weights;
  return {universal::ok_status(), api::Engine{config}};
}

api::InverseFoldDecodeOrder parse_decode_order(std::string_view text) {
  if (text == "random") {
    return api::InverseFoldDecodeOrder::Random;
  }
  if (text == "n_to_c" || text == "n-to-c") {
    return api::InverseFoldDecodeOrder::NToC;
  }
  raise_if_error(invalid_argument(
      "inverse_fold decode_order must be 'random' or 'n_to_c'"));
  return api::InverseFoldDecodeOrder::Random;
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

PyObject* logprobs_to_python(const api::InverseFoldLogProbsArtifact& artifact,
                             bool written) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  PyObjectRef shape(Py_BuildValue("(nnn)",
                                  static_cast<Py_ssize_t>(artifact.num_seqs),
                                  static_cast<Py_ssize_t>(
                                      artifact.residue_count),
                                  static_cast<Py_ssize_t>(
                                      artifact.vocab_size)));
  if (!shape) {
    return nullptr;
  }
  if (!dict.set_new_ref("path",
                        PythonCaster<std::string>::to_python(
                            artifact.path)) ||
      !dict.set_new_ref("shape", shape.release()) ||
      !dict.set_bool("written", written)) {
    return nullptr;
  }
  return dict.release();
}

PyObject* inverse_fold_result_to_python(const api::InverseFoldResult& result,
                                        bool logprobs_written) {
  ListBuilder sequences(result.sequences.size());
  if (!sequences) {
    return nullptr;
  }
  for (std::size_t index = 0; index < result.sequences.size(); ++index) {
    const api::InverseFoldSequenceResult& sequence = result.sequences[index];
    DictBuilder item;
    if (!item) {
      return nullptr;
    }
    if (!item.set_new_ref("sequence",
                          PythonCaster<std::string>::to_python(
                              sequence.sequence)) ||
        !item.set_double("score", sequence.score) ||
        !item.set_new_ref(
            "recovery_vs_native",
            PythonCaster<universal::MetricValue>::to_python(
                sequence.recovery_vs_native)) ||
        !item.set_new_ref(
            "seed",
            PyLong_FromUnsignedLongLong(
                static_cast<unsigned long long>(sequence.seed))) ||
        !item.set_new_ref("decode_order",
                          PyUnicode_FromString(
                              decode_order_name(sequence.decode_order)))) {
      return nullptr;
    }
    if (!sequences.set_new_ref(index, item.release())) {
      return nullptr;
    }
  }

  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("sequences", sequences.release()) ||
      !dict.set_new_ref("logprobs",
                        logprobs_to_python(result.logprobs,
                                           logprobs_written))) {
    return nullptr;
  }
  return dict.release();
}

PyObject* py_inverse_fold_from_structure(PyObject*,
                                         PyObject* args,
                                         PyObject* kwargs) {
  PyObject* structure_arg = nullptr;
  std::size_t num_seqs = 1;
  double sampling_temp = 0.1;
  std::size_t seed = 0;
  std::string decode_order_text = "random";
  PyObject* package = Py_None;
  double backbone_noise = 0.0;
  PyObject* out_logprobs_object = Py_None;
  std::string out_logprobs_path;
  StructureOptionFields fields;

  TypedArgParser parser(args, kwargs, "inverse_fold_from_structure");
  if (!parser.required_object("structure", structure_arg) ||
      !parser.optional("num_seqs", num_seqs, std::size_t{1}) ||
      !parser.optional("sampling_temp", sampling_temp, 0.1) ||
      !parser.optional("seed", seed, std::size_t{0}) ||
      !parse_optional_string_field(parser, "decode_order",
                                   decode_order_text) ||
      !parser.optional_object("package", package, Py_None) ||
      !parser.optional("backbone_noise", backbone_noise, 0.0) ||
      !parser.optional_object("out_logprobs", out_logprobs_object, Py_None) ||
      !parse_structure_options(parser, fields) || !parser.finish()) {
    return nullptr;
  }
  if (decode_order_text.empty()) {
    decode_order_text = "random";
  }
  const api::InverseFoldDecodeOrder decode_order =
      parse_decode_order(decode_order_text);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (!optional_string_arg(out_logprobs_object, "out_logprobs",
                           out_logprobs_path)) {
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

  const universal::Result<api::Engine> engine =
      make_inverse_fold_engine(package);
  if (raise_if_error(engine.status) < 0) {
    return nullptr;
  }

  std::string package_id;
  if (!optional_string_arg(package, "package", package_id)) {
    return nullptr;
  }
  if (package_id.empty()) {
    package_id = "proteinmpnn-v48-eps020";
  }

  api::InverseFoldRequest request{};
  request.structure = loaded.view();
  request.package = package_id;
  request.sampling_temp = static_cast<float>(sampling_temp);
  request.num_seqs = num_seqs;
  request.seed = static_cast<std::uint64_t>(seed);
  request.decode_order = decode_order;
  request.backbone_noise = static_cast<float>(backbone_noise);
  request.logprobs_out = out_logprobs_path;

  universal::Result<api::InverseFoldResult> result{universal::ok_status(), {}};
  {
    ScopedGilRelease release;
    result = engine.value.inverse_fold(request);
  }
  if (raise_if_error(result.status) < 0) {
    return nullptr;
  }

  bool logprobs_written = false;
  if (!out_logprobs_path.empty()) {
    const universal::Status status =
        io::write_inverse_fold_logprobs_npz(out_logprobs_path, result.value);
    if (raise_if_error(status) < 0) {
      return nullptr;
    }
    logprobs_written = true;
  }
  return inverse_fold_result_to_python(result.value, logprobs_written);
}

PyMethodDef kMethods[] = {
    {"inverse_fold_from_structure",
     reinterpret_cast<PyCFunction>(py_inverse_fold_from_structure),
     METH_VARARGS | METH_KEYWORDS,
     "Design sequences for one backbone structure through the public Hikoboshi "
     "API."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

int bind_inverse_fold(PyObject* module) {
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
