#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/score_alignment.hpp>
#include <hikoboshi/bindings/args.hpp>
#include <hikoboshi/bindings/casters.hpp>
#include <hikoboshi/bindings/dict_builder.hpp>
#include <hikoboshi/bindings/engine_helpers.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>
#include <hikoboshi/io/structure_loader.hpp>

#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace hikoboshi::bindings {
namespace {

class ScopedGilRelease final {
 public:
  ScopedGilRelease() : state_(PyEval_SaveThread()) {}
  ScopedGilRelease(const ScopedGilRelease&) = delete;
  ScopedGilRelease& operator=(const ScopedGilRelease&) = delete;
  ~ScopedGilRelease() { PyEval_RestoreThread(state_); }

 private:
  PyThreadState* state_;
};

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

// Decode a single correspondence step from a Python object. Accepts either a
// 2-element sequence `(q_index, t_index)` or a dict-shaped record with
// `query_index`/`target_index` integer fields. The gap sentinel value is
// `hikoboshi.ALIGNMENT_GAP_SENTINEL` (-1).
bool index_from_python(PyObject* object,
                       const char* label,
                       std::int32_t& out) {
  if (object == nullptr) {
    PyErr_Format(PyExc_TypeError, "%s must be an integer", label);
    return false;
  }
  if (!PyLong_Check(object) || PyBool_Check(object)) {
    PyErr_Format(PyExc_TypeError, "%s must be an integer", label);
    return false;
  }
  const long long value = PyLong_AsLongLong(object);
  if (value == -1 && PyErr_Occurred()) {
    return false;
  }
  if (value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    PyErr_Format(PyExc_OverflowError, "%s does not fit in int32", label);
    return false;
  }
  out = static_cast<std::int32_t>(value);
  return true;
}

bool decode_step_from_sequence(PyObject* item,
                               universal::AlignmentStep& step) {
  PyObjectRef sequence(PySequence_Fast(
      item, "correspondence step must be a 2-element sequence or dict"));
  if (!sequence) {
    return false;
  }
  const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.get());
  if (size != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "correspondence step must contain exactly two indices");
    return false;
  }
  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  return index_from_python(items[0], "query_index", step.query_index) &&
         index_from_python(items[1], "target_index", step.target_index);
}

bool decode_step_from_dict(PyObject* item,
                           universal::AlignmentStep& step) {
  PyObject* query = PyDict_GetItemString(item, "query_index");
  PyObject* target = PyDict_GetItemString(item, "target_index");
  if (query == nullptr || target == nullptr) {
    PyErr_SetString(
        PyExc_KeyError,
        "correspondence dict must define 'query_index' and 'target_index'");
    return false;
  }
  return index_from_python(query, "query_index", step.query_index) &&
         index_from_python(target, "target_index", step.target_index);
}

bool decode_correspondences(PyObject* object,
                            universal::AlignmentPath& out) {
  out.steps.clear();
  out.aligned_pairs = 0;
  if (object == nullptr || object == Py_None) {
    return true;
  }
  PyObjectRef sequence(
      PySequence_Fast(object, "correspondences must be a sequence"));
  if (!sequence) {
    return false;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());
  out.steps.reserve(static_cast<std::size_t>(count));
  PyObject** items = PySequence_Fast_ITEMS(sequence.get());
  for (Py_ssize_t index = 0; index < count; ++index) {
    universal::AlignmentStep step{};
    step.residue_score = 0.0F;
    PyObject* item = items[index];
    if (PyDict_Check(item)) {
      if (!decode_step_from_dict(item, step)) {
        return false;
      }
    } else {
      if (!decode_step_from_sequence(item, step)) {
        return false;
      }
    }
    out.steps.push_back(step);
    if (step.query_index != universal::kAlignmentGapSentinel &&
        step.target_index != universal::kAlignmentGapSentinel) {
      ++out.aligned_pairs;
    }
  }
  return true;
}

PyObject* score_alignment_result_to_python(
    const api::ScoreAlignmentResult& result) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  using PythonMetric = PythonCaster<universal::MetricValue>;
  if (!dict.set_new_ref("rmsd", PythonMetric::to_python(result.rmsd)) ||
      !dict.set_new_ref("tm_score_query",
                        PythonMetric::to_python(result.tm_score_query)) ||
      !dict.set_new_ref("tm_score_target",
                        PythonMetric::to_python(result.tm_score_target)) ||
      !dict.set_new_ref("lddt", PythonMetric::to_python(result.lddt)) ||
      !dict.set_new_ref("lddt_byA",
                        PythonMetric::to_python(result.lddt_byA)) ||
      !dict.set_new_ref("lddt_byB",
                        PythonMetric::to_python(result.lddt_byB)) ||
      !dict.set_new_ref("lddt_aln",
                        PythonMetric::to_python(result.lddt_aln)) ||
      !dict.set_new_ref("identity",
                        PythonMetric::to_python(result.identity)) ||
      !dict.set_new_ref("coverage_query",
                        PythonMetric::to_python(result.coverage_query)) ||
      !dict.set_new_ref("coverage_target",
                        PythonMetric::to_python(result.coverage_target)) ||
      !dict.set_new_ref("coverage_mean",
                        PythonMetric::to_python(result.coverage_mean)) ||
      !dict.set_new_ref("coverage_byA",
                        PythonMetric::to_python(result.coverage_byA)) ||
      !dict.set_new_ref("coverage_byB",
                        PythonMetric::to_python(result.coverage_byB)) ||
      !dict.set_new_ref("ecs", PythonMetric::to_python(result.ecs)) ||
      !dict.set_size("aligned_pairs", result.aligned_pairs)) {
    return nullptr;
  }
  return dict.release();
}

PyObject* py_score_alignment_from_structure(PyObject*,
                                            PyObject* args,
                                            PyObject* kwargs) {
  PyObject* query_arg = nullptr;
  PyObject* target_arg = nullptr;
  PyObject* correspondences_arg = nullptr;
  StructureOptionFields fields;
  TypedArgParser parser(args, kwargs, "score_alignment_from_structure");
  if (!parser.required_object("query", query_arg) ||
      !parser.required_object("target", target_arg) ||
      !parser.required_object("correspondences", correspondences_arg) ||
      !parse_structure_options(parser, fields) || !parser.finish()) {
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

  api::ScoreAlignmentRequest request{};
  request.query_structure = query_loaded.view();
  request.target_structure = target_loaded.view();
  if (!decode_correspondences(correspondences_arg, request.correspondences)) {
    return nullptr;
  }

  api::ScoreAlignmentResult result{};
  universal::Status status{};
  {
    ScopedGilRelease gil_release;
    status = api::score_alignment(request, result);
  }
  if (raise_if_error(status) < 0) {
    return nullptr;
  }
  return score_alignment_result_to_python(result);
}

PyMethodDef kMethods[] = {
    {"score_alignment_from_structure",
     reinterpret_cast<PyCFunction>(py_score_alignment_from_structure),
     METH_VARARGS | METH_KEYWORDS,
     "Score an externally-supplied alignment over two structure paths. "
     "Returns the same metric panel pairwise() returns, computed without "
     "running Hikoboshi's own alignment. correspondences is a sequence of "
     "(query_index, target_index) pairs using zero-based residue indices; "
     "use hikoboshi.ALIGNMENT_GAP_SENTINEL (-1) on either side to mark a gap "
     "step."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

int bind_score_alignment(PyObject* module) {
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
