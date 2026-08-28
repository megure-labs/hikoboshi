#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/results.hpp>
#include <hikoboshi/bindings/casters.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_b = hikoboshi::bindings;
namespace hiko_u = hikoboshi::universal;

namespace {

template <typename T, typename = void>
struct HasToPython : std::false_type {};

template <typename T>
struct HasToPython<
    T,
    std::void_t<decltype(
        hiko_b::PythonCaster<T>::to_python(std::declval<const T&>()))>>
    : std::true_type {};

bool fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  return false;
}

bool require(bool condition, const char* message) {
  return condition ? true : fail(message);
}

PyObject* dict_item(PyObject* dict, const char* key) {
  return PyDict_Check(dict) ? PyDict_GetItemString(dict, key) : nullptr;
}

bool unicode_equals(PyObject* object, const char* expected) {
  if (!PyUnicode_Check(object)) {
    return false;
  }
  const char* actual = PyUnicode_AsUTF8(object);
  return actual != nullptr && std::string_view{actual} == expected;
}

bool long_equals(PyObject* object, long expected) {
  if (!PyLong_Check(object)) {
    return false;
  }
  const long actual = PyLong_AsLong(object);
  return !(actual == -1 && PyErr_Occurred()) && actual == expected;
}

bool double_near(PyObject* object, double expected) {
  if (!PyFloat_Check(object)) {
    return false;
  }
  return std::fabs(PyFloat_AsDouble(object) - expected) < 1.0e-6;
}

bool test_scalar_and_enum_casters() {
  std::int32_t parsed_i32 = 0;
  hiko_b::PyObjectRef py_i32(PyLong_FromLong(42));
  if (!require(hiko_b::PythonCaster<std::int32_t>::from_python(py_i32.get(),
                                                            parsed_i32),
               "failed to parse int32") ||
      !require(parsed_i32 == 42, "parsed wrong int32 value")) {
    return false;
  }

  hiko_b::PyObjectRef bad_int(PyUnicode_FromString("42"));
  if (!require(!hiko_b::PythonCaster<std::int32_t>::from_python(bad_int.get(),
                                                            parsed_i32),
               "string unexpectedly parsed as int32") ||
      !require(PyErr_ExceptionMatches(PyExc_TypeError),
               "bad int32 parse raised wrong exception")) {
    return false;
  }
  PyErr_Clear();

  hiko_b::PyObjectRef status(
      hiko_b::PythonCaster<hiko_u::StatusCode>::to_python(
          hiko_u::StatusCode::InvalidArgument));
  if (!require(unicode_equals(status.get(), "invalid_argument"),
               "status code did not render to chartered string")) {
    return false;
  }

  hiko_u::MetricInvalidReason reason = hiko_u::MetricInvalidReason::None;
  hiko_b::PyObjectRef py_reason(PyUnicode_FromString("missing_structure_metadata"));
  if (!require(hiko_b::PythonCaster<hiko_u::MetricInvalidReason>::from_python(
                   py_reason.get(), reason),
               "failed to parse metric invalid reason") ||
      !require(reason == hiko_u::MetricInvalidReason::MissingStructureMetadata,
               "parsed wrong metric invalid reason")) {
    return false;
  }
  return true;
}

bool test_metric_and_path_casters() {
  const hiko_u::MetricValue invalid_metric{
      0.0, false, hiko_u::MetricInvalidReason::MissingStructureMetadata};
  hiko_b::PyObjectRef py_metric(
      hiko_b::PythonCaster<hiko_u::MetricValue>::to_python(invalid_metric));
  if (!require(static_cast<bool>(py_metric) && PyDict_Check(py_metric.get()),
               "metric caster did not return a dict") ||
      !require(dict_item(py_metric.get(), "valid") == Py_False,
               "invalid metric did not carry valid=false") ||
      !require(unicode_equals(dict_item(py_metric.get(), "reason"),
                              "missing_structure_metadata"),
               "metric invalid reason did not render")) {
    return false;
  }

  hiko_u::AlignmentPath path{};
  path.steps.push_back({0, 1, 2.5F});
  path.aligned_pairs = 1;
  path.query_start = 0;
  path.query_end = 0;
  path.target_start = 1;
  path.target_end = 1;
  hiko_b::PyObjectRef py_path(hiko_b::PythonCaster<hiko_u::AlignmentPath>::to_python(path));
  PyObject* steps = dict_item(py_path.get(), "steps");
  PyObject* first_step =
      PyList_Check(steps) && PyList_GET_SIZE(steps) == 1
          ? PyList_GET_ITEM(steps, 0)
          : nullptr;
  if (!require(first_step != nullptr, "alignment path steps missing") ||
      !require(long_equals(dict_item(first_step, "query_index"), 0),
               "alignment step query index mismatch") ||
      !require(long_equals(dict_item(first_step, "target_index"), 1),
               "alignment step target index mismatch") ||
      !require(double_near(dict_item(first_step, "residue_score"), 2.5),
               "alignment step score mismatch") ||
      !require(long_equals(dict_item(py_path.get(), "aligned_pairs"), 1),
               "alignment path aligned pair count mismatch")) {
    return false;
  }
  return true;
}

bool test_pairwise_result_caster() {
  hiko::PairwiseResult result{};
  result.path.steps.push_back({0, 0, 3.0F});
  result.path.aligned_pairs = 1;
  result.path.query_start = 0;
  result.path.query_end = 0;
  result.path.target_start = 0;
  result.path.target_end = 0;
  result.metrics.raw_sw_score = 3.0;
  result.metrics.coverage_query = {1.0, true, hiko_u::MetricInvalidReason::None};
  result.metrics.rmsd = {
      0.0, false, hiko_u::MetricInvalidReason::MissingStructureMetadata};
  result.warnings.push_back({
      hiko_u::PackageWarningKind::GapDefaultsOverridden,
      hiko_u::PackageValidationStage::GapModelDefaults,
      "gap_defaults_overridden",
      "Gap defaults were overridden",
  });

  hiko_b::PyObjectRef py_result(
      hiko_b::PythonCaster<hiko::PairwiseResult>::to_python(result));
  PyObject* metrics = dict_item(py_result.get(), "metrics");
  PyObject* warnings = dict_item(py_result.get(), "warnings");
  PyObject* first_warning =
      PyList_Check(warnings) && PyList_GET_SIZE(warnings) == 1
          ? PyList_GET_ITEM(warnings, 0)
          : nullptr;
  if (!require(static_cast<bool>(py_result) && PyDict_Check(py_result.get()),
               "pairwise result caster did not return a dict") ||
      !require(double_near(dict_item(metrics, "raw_sw_score"), 3.0),
               "pairwise metric raw score mismatch") ||
      !require(unicode_equals(dict_item(dict_item(metrics, "rmsd"), "reason"),
                              "missing_structure_metadata"),
               "pairwise invalid metric reason mismatch") ||
      !require(first_warning != nullptr, "pairwise warning missing") ||
      !require(unicode_equals(dict_item(first_warning, "kind"),
                              "gap_defaults_overridden"),
               "warning kind mismatch") ||
      !require(unicode_equals(dict_item(first_warning, "stage"),
                              "gap_model_defaults"),
               "warning stage mismatch")) {
    return false;
  }
  return true;
}

bool test_closed_family_compile_shape() {
  static_assert(HasToPython<std::int32_t>::value,
                "int32 must have an explicit Python caster");
  static_assert(HasToPython<hiko_u::AlignmentPath>::value,
                "alignment path must have an explicit Python caster");
  static_assert(!HasToPython<std::vector<int>>::value,
                "vector catch-all Python caster must not exist");
  return true;
}

}  // namespace

int main() {
  Py_Initialize();

  const bool ok = test_closed_family_compile_shape() &&
                  test_scalar_and_enum_casters() &&
                  test_metric_and_path_casters() &&
                  test_pairwise_result_caster();

  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  const int finalize_status = Py_FinalizeEx();
  return ok && finalize_status == 0 ? 0 : 1;
}
