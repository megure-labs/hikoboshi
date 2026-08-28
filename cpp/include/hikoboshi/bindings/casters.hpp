#ifndef HIKOBOSHI_BINDINGS_CASTERS_HPP
#define HIKOBOSHI_BINDINGS_CASTERS_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/results.hpp>
#include <hikoboshi/bindings/dict_builder.hpp>
#include <hikoboshi/bindings/list_builder.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>
#include <hikoboshi/bindings/tensor_type.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace hikoboshi::bindings {

namespace detail {

template <typename T>
struct PythonCasterTraits {};

}  // namespace detail

template <typename T>
struct PythonCaster : detail::PythonCasterTraits<T> {};

namespace detail {

inline PyObject* string_view_to_python(std::string_view value) {
  return PyUnicode_FromStringAndSize(
      value.data() == nullptr ? "" : value.data(),
      static_cast<Py_ssize_t>(value.size()));
}

inline bool unicode_to_string(PyObject* object,
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

inline bool string_eq(std::string_view lhs, const char* rhs) noexcept {
  return lhs == std::string_view{rhs};
}

template <typename T>
bool signed_long_from_python(PyObject* object, const char* label, T& out) {
  static_assert(std::is_signed<T>::value, "T must be signed");
  if (!PyLong_Check(object) || PyBool_Check(object)) {
    PyErr_Format(PyExc_TypeError, "%s must be an integer", label);
    return false;
  }
  const long long value = PyLong_AsLongLong(object);
  if (value == -1 && PyErr_Occurred()) {
    return false;
  }
  if (value < static_cast<long long>(std::numeric_limits<T>::min()) ||
      value > static_cast<long long>(std::numeric_limits<T>::max())) {
    PyErr_Format(PyExc_OverflowError, "%s is outside the supported range",
                 label);
    return false;
  }
  out = static_cast<T>(value);
  return true;
}

template <typename T>
bool unsigned_long_from_python(PyObject* object, const char* label, T& out) {
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");
  if (!PyLong_Check(object) || PyBool_Check(object)) {
    PyErr_Format(PyExc_TypeError, "%s must be an unsigned integer", label);
    return false;
  }
  const unsigned long long value = PyLong_AsUnsignedLongLong(object);
  if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
    return false;
  }
  if (value >
      static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
    PyErr_Format(PyExc_OverflowError, "%s is outside the supported range",
                 label);
    return false;
  }
  out = static_cast<T>(value);
  return true;
}

inline const char* status_code_name(
    universal::StatusCode code) noexcept {
  switch (code) {
    case universal::StatusCode::Ok:
      return "ok";
    case universal::StatusCode::InvalidArgument:
      return "invalid_argument";
    case universal::StatusCode::FailedPrecondition:
      return "failed_precondition";
    case universal::StatusCode::Unavailable:
      return "unavailable";
    case universal::StatusCode::Unimplemented:
      return "unimplemented";
    case universal::StatusCode::InternalError:
      return "internal_error";
  }
  return "unknown_status";
}

inline bool status_code_from_name(std::string_view name,
                                  universal::StatusCode& out) noexcept {
  if (string_eq(name, "ok")) {
    out = universal::StatusCode::Ok;
  } else if (string_eq(name, "invalid_argument")) {
    out = universal::StatusCode::InvalidArgument;
  } else if (string_eq(name, "failed_precondition")) {
    out = universal::StatusCode::FailedPrecondition;
  } else if (string_eq(name, "unavailable")) {
    out = universal::StatusCode::Unavailable;
  } else if (string_eq(name, "unimplemented")) {
    out = universal::StatusCode::Unimplemented;
  } else if (string_eq(name, "internal_error")) {
    out = universal::StatusCode::InternalError;
  } else {
    return false;
  }
  return true;
}

inline const char* metric_reason_name(
    universal::MetricInvalidReason reason) noexcept {
  switch (reason) {
    case universal::MetricInvalidReason::None:
      return "none";
    case universal::MetricInvalidReason::Unavailable:
      return "unavailable";
    case universal::MetricInvalidReason::MissingSequenceMetadata:
      return "missing_sequence_metadata";
    case universal::MetricInvalidReason::MissingStructureMetadata:
      return "missing_structure_metadata";
    case universal::MetricInvalidReason::InsufficientAlignedPairs:
      return "insufficient_aligned_pairs";
    case universal::MetricInvalidReason::ZeroDenominator:
      return "zero_denominator";
    case universal::MetricInvalidReason::Unimplemented:
      return "unimplemented";
  }
  return "unknown_metric_invalid_reason";
}

inline bool metric_reason_from_name(
    std::string_view name,
    universal::MetricInvalidReason& out) noexcept {
  if (string_eq(name, "none")) {
    out = universal::MetricInvalidReason::None;
  } else if (string_eq(name, "unavailable")) {
    out = universal::MetricInvalidReason::Unavailable;
  } else if (string_eq(name, "missing_sequence_metadata")) {
    out = universal::MetricInvalidReason::MissingSequenceMetadata;
  } else if (string_eq(name, "missing_structure_metadata")) {
    out = universal::MetricInvalidReason::MissingStructureMetadata;
  } else if (string_eq(name, "insufficient_aligned_pairs")) {
    out = universal::MetricInvalidReason::InsufficientAlignedPairs;
  } else if (string_eq(name, "zero_denominator")) {
    out = universal::MetricInvalidReason::ZeroDenominator;
  } else if (string_eq(name, "unimplemented")) {
    out = universal::MetricInvalidReason::Unimplemented;
  } else {
    return false;
  }
  return true;
}

inline const char* warning_kind_name(
    universal::PackageWarningKind kind) noexcept {
  switch (kind) {
    case universal::PackageWarningKind::Unspecified:
      return "unspecified";
    case universal::PackageWarningKind::GapDefaultsOverridden:
      return "gap_defaults_overridden";
    case universal::PackageWarningKind::UnsupportedReservedCapability:
      return "unsupported_reserved_capability";
    case universal::PackageWarningKind::IgnoredHistoricalTensor:
      return "ignored_historical_tensor";
  }
  return "unspecified";
}

inline const char* validation_stage_name(
    universal::PackageValidationStage stage) noexcept {
  switch (stage) {
    case universal::PackageValidationStage::SchemaVersion:
      return "schema_version";
    case universal::PackageValidationStage::StorageChecksum:
      return "storage_checksum";
    case universal::PackageValidationStage::ArchitectureRegistration:
      return "architecture_registration";
    case universal::PackageValidationStage::TensorTableRolesShapesDtypes:
      return "tensor_table_roles_shapes_dtypes";
    case universal::PackageValidationStage::InputRoute:
      return "input_route";
    case universal::PackageValidationStage::PreprocessingCapabilities:
      return "preprocessing_capabilities";
    case universal::PackageValidationStage::ScoringMethod:
      return "scoring_method";
    case universal::PackageValidationStage::ScoreMatrixSemantics:
      return "score_matrix_semantics";
    case universal::PackageValidationStage::GapModelDefaults:
      return "gap_model_defaults";
    case universal::PackageValidationStage::AlignmentAlgorithm:
      return "alignment_algorithm";
    case universal::PackageValidationStage::WorkflowCompatibility:
      return "workflow_compatibility";
    case universal::PackageValidationStage::PreparedStateBuild:
      return "prepared_state_build";
  }
  return "workflow_compatibility";
}

inline PyObject* none_to_python() {
  Py_INCREF(Py_None);
  return Py_None;
}

inline PyObject* residue_codes_to_python(const std::vector<char>& codes) {
  return PyUnicode_FromStringAndSize(codes.empty() ? "" : codes.data(),
                                     static_cast<Py_ssize_t>(codes.size()));
}

inline PyObject* float_matrix_to_python(const std::vector<float>& values,
                                        std::size_t row_count,
                                        std::size_t column_count) {
  const std::size_t expected = row_count * column_count;
  if (values.size() < expected) {
    PyErr_SetString(PyExc_ValueError, "encoded embedding payload is incomplete");
    return nullptr;
  }

  ListBuilder rows(row_count);
  if (!rows) {
    return nullptr;
  }
  for (std::size_t row_index = 0; row_index < row_count; ++row_index) {
    ListBuilder row(column_count);
    if (!row) {
      return nullptr;
    }
    const std::size_t row_offset = row_index * column_count;
    for (std::size_t column_index = 0; column_index < column_count;
         ++column_index) {
      if (!row.set_new_ref(
              column_index,
              PyFloat_FromDouble(values[row_offset + column_index]))) {
        return nullptr;
      }
    }
    if (!rows.set_new_ref(row_index, row.release())) {
      return nullptr;
    }
  }
  return rows.release();
}

inline PyObject* metric_to_python(const universal::MetricValue& metric);
inline PyObject* residue_metadata_to_python(
    const universal::ResidueMetadataView& residue);
inline PyObject* warnings_to_python(
    const std::vector<universal::PackageWarning>& warnings);

}  // namespace detail

template <>
struct detail::PythonCasterTraits<bool> {
  static PyObject* to_python(bool value) {
    return PyBool_FromLong(value ? 1 : 0);
  }

  static bool from_python(PyObject* object, bool& out) {
    if (!PyBool_Check(object)) {
      PyErr_SetString(PyExc_TypeError, "value must be a bool");
      return false;
    }
    out = object == Py_True;
    return true;
  }
};

template <>
struct detail::PythonCasterTraits<std::int32_t> {
  static PyObject* to_python(std::int32_t value) {
    return PyLong_FromLong(static_cast<long>(value));
  }

  static bool from_python(PyObject* object, std::int32_t& out) {
    return detail::signed_long_from_python(object, "value", out);
  }
};

template <>
struct detail::PythonCasterTraits<std::int64_t> {
  static PyObject* to_python(std::int64_t value) {
    return PyLong_FromLongLong(static_cast<long long>(value));
  }

  static bool from_python(PyObject* object, std::int64_t& out) {
    return detail::signed_long_from_python(object, "value", out);
  }
};

template <>
struct detail::PythonCasterTraits<std::size_t> {
  static PyObject* to_python(std::size_t value) {
    return PyLong_FromSize_t(value);
  }

  static bool from_python(PyObject* object, std::size_t& out) {
    return detail::unsigned_long_from_python(object, "value", out);
  }
};

template <>
struct detail::PythonCasterTraits<float> {
  static PyObject* to_python(float value) {
    return PyFloat_FromDouble(static_cast<double>(value));
  }

  static bool from_python(PyObject* object, float& out) {
    if (PyBool_Check(object)) {
      PyErr_SetString(PyExc_TypeError, "value must be a real number");
      return false;
    }
    const double value = PyFloat_AsDouble(object);
    if (value == -1.0 && PyErr_Occurred()) {
      return false;
    }
    if (value < -std::numeric_limits<float>::max() ||
        value > std::numeric_limits<float>::max()) {
      PyErr_SetString(PyExc_OverflowError, "value is outside float range");
      return false;
    }
    out = static_cast<float>(value);
    return true;
  }
};

template <>
struct detail::PythonCasterTraits<double> {
  static PyObject* to_python(double value) { return PyFloat_FromDouble(value); }

  static bool from_python(PyObject* object, double& out) {
    if (PyBool_Check(object)) {
      PyErr_SetString(PyExc_TypeError, "value must be a real number");
      return false;
    }
    const double value = PyFloat_AsDouble(object);
    if (value == -1.0 && PyErr_Occurred()) {
      return false;
    }
    out = value;
    return true;
  }
};

template <>
struct detail::PythonCasterTraits<std::string> {
  static PyObject* to_python(const std::string& value) {
    return PyUnicode_FromStringAndSize(value.data(),
                                       static_cast<Py_ssize_t>(value.size()));
  }

  static bool from_python(PyObject* object, std::string& out) {
    return detail::unicode_to_string(object, "value", out);
  }
};

template <>
struct detail::PythonCasterTraits<std::string_view> {
  static PyObject* to_python(std::string_view value) {
    return detail::string_view_to_python(value);
  }
};

template <>
struct detail::PythonCasterTraits<universal::StatusCode> {
  static PyObject* to_python(universal::StatusCode code) {
    return PyUnicode_FromString(detail::status_code_name(code));
  }

  static bool from_python(PyObject* object, universal::StatusCode& out) {
    std::string name;
    if (!detail::unicode_to_string(object, "status code", name)) {
      return false;
    }
    if (!detail::status_code_from_name(name, out)) {
      PyErr_SetString(PyExc_ValueError, "unknown Hikoboshi status code");
      return false;
    }
    return true;
  }
};

template <>
struct detail::PythonCasterTraits<universal::Status> {
  static PyObject* to_python(const universal::Status& status) {
    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("code",
                          PythonCaster<universal::StatusCode>::to_python(
                              status.code)) ||
        !dict.set_new_ref("detail",
                          detail::string_view_to_python(
                              status.detail == nullptr ? std::string_view{}
                                                       : status.detail))) {
      return nullptr;
    }
    return dict.release();
  }
};

template <>
struct detail::PythonCasterTraits<universal::MetricInvalidReason> {
  static PyObject* to_python(universal::MetricInvalidReason reason) {
    return PyUnicode_FromString(detail::metric_reason_name(reason));
  }

  static bool from_python(PyObject* object,
                          universal::MetricInvalidReason& out) {
    std::string name;
    if (!detail::unicode_to_string(object, "metric invalid reason", name)) {
      return false;
    }
    if (!detail::metric_reason_from_name(name, out)) {
      PyErr_SetString(PyExc_ValueError,
                      "unknown Hikoboshi metric invalid reason");
      return false;
    }
    return true;
  }
};

template <>
struct detail::PythonCasterTraits<universal::MetricValue> {
  static PyObject* to_python(const universal::MetricValue& metric) {
    return detail::metric_to_python(metric);
  }

  static bool from_python(PyObject* object, universal::MetricValue& out) {
    if (!PyDict_Check(object)) {
      PyErr_SetString(PyExc_TypeError, "metric value must be a dict");
      return false;
    }
    PyObject* value = PyDict_GetItemString(object, "value");
    PyObject* valid = PyDict_GetItemString(object, "valid");
    PyObject* reason = PyDict_GetItemString(object, "reason");
    if (value == nullptr || valid == nullptr || reason == nullptr) {
      PyErr_SetString(PyExc_KeyError,
                      "metric value needs value, valid, and reason");
      return false;
    }
    return PythonCaster<double>::from_python(value, out.value) &&
           PythonCaster<bool>::from_python(valid, out.valid) &&
           PythonCaster<universal::MetricInvalidReason>::from_python(
               reason, out.reason);
  }
};

template <>
struct detail::PythonCasterTraits<universal::AlignmentStep> {
  static PyObject* to_python(const universal::AlignmentStep& step) {
    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("query_index",
                          PythonCaster<std::int32_t>::to_python(
                              step.query_index)) ||
        !dict.set_new_ref("target_index",
                          PythonCaster<std::int32_t>::to_python(
                              step.target_index)) ||
        !dict.set_new_ref("residue_score",
                          PythonCaster<float>::to_python(
                              step.residue_score))) {
      return nullptr;
    }
    return dict.release();
  }

  static bool from_python(PyObject* object, universal::AlignmentStep& out) {
    if (!PyDict_Check(object)) {
      PyErr_SetString(PyExc_TypeError, "alignment step must be a dict");
      return false;
    }
    PyObject* query = PyDict_GetItemString(object, "query_index");
    PyObject* target = PyDict_GetItemString(object, "target_index");
    PyObject* score = PyDict_GetItemString(object, "residue_score");
    if (query == nullptr || target == nullptr || score == nullptr) {
      PyErr_SetString(
          PyExc_KeyError,
          "alignment step needs query_index, target_index, and residue_score");
      return false;
    }
    return PythonCaster<std::int32_t>::from_python(query, out.query_index) &&
           PythonCaster<std::int32_t>::from_python(target, out.target_index) &&
           PythonCaster<float>::from_python(score, out.residue_score);
  }
};

template <>
struct detail::PythonCasterTraits<universal::AlignmentPath> {
  static PyObject* to_python(const universal::AlignmentPath& path) {
    ListBuilder steps(path.steps.size());
    if (!steps) {
      return nullptr;
    }
    for (std::size_t index = 0; index < path.steps.size(); ++index) {
      if (!steps.set_new_ref(
              index,
              PythonCaster<universal::AlignmentStep>::to_python(
                  path.steps[index]))) {
        return nullptr;
      }
    }

    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("steps", steps.release()) ||
        !dict.set_new_ref("aligned_pairs",
                          PythonCaster<std::size_t>::to_python(
                              path.aligned_pairs)) ||
        !dict.set_new_ref("query_start",
                          PythonCaster<std::int32_t>::to_python(
                              path.query_start)) ||
        !dict.set_new_ref("query_end",
                          PythonCaster<std::int32_t>::to_python(
                              path.query_end)) ||
        !dict.set_new_ref("target_start",
                          PythonCaster<std::int32_t>::to_python(
                              path.target_start)) ||
        !dict.set_new_ref("target_end",
                          PythonCaster<std::int32_t>::to_python(
                              path.target_end))) {
      return nullptr;
    }
    return dict.release();
  }
};

template <>
struct detail::PythonCasterTraits<universal::ResidueMetadataView> {
  static PyObject* to_python(const universal::ResidueMetadataView& residue) {
    return detail::residue_metadata_to_python(residue);
  }
};

template <>
struct detail::PythonCasterTraits<universal::PackageWarningKind> {
  static PyObject* to_python(universal::PackageWarningKind kind) {
    return PyUnicode_FromString(detail::warning_kind_name(kind));
  }
};

template <>
struct detail::PythonCasterTraits<universal::PackageValidationStage> {
  static PyObject* to_python(universal::PackageValidationStage stage) {
    return PyUnicode_FromString(detail::validation_stage_name(stage));
  }
};

template <>
struct detail::PythonCasterTraits<universal::PackageWarning> {
  static PyObject* to_python(const universal::PackageWarning& warning) {
    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("kind",
                          PythonCaster<universal::PackageWarningKind>::
                              to_python(warning.kind)) ||
        !dict.set_new_ref("stage",
                          PythonCaster<universal::PackageValidationStage>::
                              to_python(warning.stage)) ||
        !dict.set_new_ref("code",
                          detail::string_view_to_python(warning.code)) ||
        !dict.set_new_ref("message",
                          detail::string_view_to_python(warning.message))) {
      return nullptr;
    }
    return dict.release();
  }
};

template <>
struct detail::PythonCasterTraits<api::PairwiseMetrics> {
  static PyObject* to_python(const api::PairwiseMetrics& metrics) {
    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("raw_sw_score",
                          PythonCaster<double>::to_python(
                              metrics.raw_sw_score)) ||
        !dict.set_new_ref("soft_sw_score",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.soft_sw_score)) ||
        !dict.set_new_ref("coverage_query",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.coverage_query)) ||
        !dict.set_new_ref("coverage_target",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.coverage_target)) ||
        !dict.set_new_ref("coverage_mean",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.coverage_mean)) ||
        !dict.set_new_ref("identity",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.identity)) ||
        !dict.set_new_ref("rmsd",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.rmsd)) ||
        !dict.set_new_ref("tm_score_query",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.tm_score_query)) ||
        !dict.set_new_ref("tm_score_target",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.tm_score_target)) ||
        !dict.set_new_ref("lddt",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.lddt)) ||
        !dict.set_new_ref("lddt_byA",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.lddt_byA)) ||
        !dict.set_new_ref("lddt_byB",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.lddt_byB)) ||
        !dict.set_new_ref("lddt_aln",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.lddt_aln)) ||
        !dict.set_new_ref("coverage_byA",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.coverage_byA)) ||
        !dict.set_new_ref("coverage_byB",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.coverage_byB)) ||
        !dict.set_new_ref("ecs",
                          PythonCaster<universal::MetricValue>::to_python(
                              metrics.ecs))) {
      return nullptr;
    }
    return dict.release();
  }
};

namespace detail {

inline universal::MetricValue invalid_metric(
    universal::MetricInvalidReason reason) noexcept {
  return {0.0, false, reason};
}

inline universal::MetricValue valid_metric(double value) noexcept {
  return {value, true, universal::MetricInvalidReason::None};
}

inline universal::MetricValue sw_per_aligned_metric(
    const api::PairwiseResult& result) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return invalid_metric(universal::MetricInvalidReason::ZeroDenominator);
  }
  return valid_metric(result.metrics.raw_sw_score /
                      static_cast<double>(result.path.aligned_pairs));
}

inline universal::MetricValue sw_per_length_metric(
    const api::PairwiseResult& result,
    universal::MetricValue coverage) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return invalid_metric(universal::MetricInvalidReason::ZeroDenominator);
  }
  if (!coverage.valid) {
    return invalid_metric(coverage.reason);
  }
  return valid_metric((result.metrics.raw_sw_score * coverage.value) /
                      static_cast<double>(result.path.aligned_pairs));
}

inline bool set_metric_item(PyObject* dict,
                            const char* key,
                            universal::MetricValue metric) {
  PyObjectRef value(PythonCaster<universal::MetricValue>::to_python(metric));
  if (!value) {
    return false;
  }
  return PyDict_SetItemString(dict, key, value.get()) == 0;
}

inline PyObject* pairwise_metrics_to_python(
    const api::PairwiseResult& result) {
  PyObjectRef metrics(PythonCaster<api::PairwiseMetrics>::to_python(
      result.metrics));
  if (!metrics) {
    return nullptr;
  }
  if (!set_metric_item(metrics.get(), "sw_per_query_len",
                       sw_per_length_metric(
                           result, result.metrics.coverage_query)) ||
      !set_metric_item(metrics.get(), "sw_per_target_len",
                       sw_per_length_metric(
                           result, result.metrics.coverage_target)) ||
      !set_metric_item(metrics.get(), "sw_per_aligned",
                       sw_per_aligned_metric(result))) {
    return nullptr;
  }
  return metrics.release();
}

}  // namespace detail

template <>
struct detail::PythonCasterTraits<api::EncodedEmbedding> {
  static PyObject* to_python(const api::EncodedEmbedding& embedding) {
    return tensor_from_float32_matrix(embedding.values, embedding.residue_count,
                                      embedding.dimension);
  }
};

template <>
struct detail::PythonCasterTraits<api::EncodeResult> {
  static PyObject* to_python(const api::EncodeResult& result) {
    ListBuilder residues(result.embedding.residues.size());
    if (!residues) {
      return nullptr;
    }
    for (std::size_t index = 0; index < result.embedding.residues.size();
         ++index) {
      if (!residues.set_new_ref(
              index,
              PythonCaster<universal::ResidueMetadataView>::to_python(
                  result.embedding.residues[index]))) {
        return nullptr;
      }
    }

    DictBuilder metadata;
    if (!metadata) {
      return nullptr;
    }
    if (!metadata.set_new_ref("residue_count",
                              PythonCaster<std::size_t>::to_python(
                                  result.embedding.residue_count)) ||
        !metadata.set_new_ref("dimension",
                              PythonCaster<std::size_t>::to_python(
                                  result.embedding.dimension)) ||
        !metadata.set_new_ref(
            "residue_codes",
            detail::residue_codes_to_python(result.embedding.residue_codes)) ||
        !metadata.set_new_ref("residues", residues.release())) {
      return nullptr;
    }

    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref(
            "embeddings",
            PythonCaster<api::EncodedEmbedding>::to_python(
                result.embedding)) ||
        !dict.set_new_ref("metadata", metadata.release())) {
      return nullptr;
    }
    return dict.release();
  }
};

template <>
struct detail::PythonCasterTraits<api::PairwiseResult> {
  static PyObject* to_python(const api::PairwiseResult& result) {
    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("path",
                          PythonCaster<api::AlignmentPath>::to_python(
                              result.path)) ||
        !dict.set_new_ref("metrics",
                          detail::pairwise_metrics_to_python(result)) ||
        !dict.set_new_ref("warnings",
                          detail::warnings_to_python(result.warnings))) {
      return nullptr;
    }
    return dict.release();
  }
};

template <>
struct detail::PythonCasterTraits<api::PairwiseResultRecord> {
  static PyObject* to_python(const api::PairwiseResultRecord& record) {
    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("query_index",
                          PythonCaster<std::size_t>::to_python(
                              record.query_index)) ||
        !dict.set_new_ref("target_index",
                          PythonCaster<std::size_t>::to_python(
                              record.target_index)) ||
        !dict.set_new_ref("result",
                          PythonCaster<api::PairwiseResult>::to_python(
                              record.result))) {
      return nullptr;
    }
    return dict.release();
  }
};

template <>
struct detail::PythonCasterTraits<api::AllVsAllResult> {
  static PyObject* to_python(const api::AllVsAllResult& result) {
    ListBuilder records(result.records.size());
    if (!records) {
      return nullptr;
    }
    for (std::size_t index = 0; index < result.records.size(); ++index) {
      if (!records.set_new_ref(
              index,
              PythonCaster<api::PairwiseResultRecord>::to_python(
                  result.records[index]))) {
        return nullptr;
      }
    }

    DictBuilder dict;
    if (!dict) {
      return nullptr;
    }
    if (!dict.set_new_ref("records", records.release())) {
      return nullptr;
    }
    return dict.release();
  }
};

namespace detail {

inline PyObject* metric_to_python(const universal::MetricValue& metric) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("value", PythonCaster<double>::to_python(metric.value)) ||
      !dict.set_new_ref("valid", PythonCaster<bool>::to_python(metric.valid)) ||
      !dict.set_new_ref(
          "reason",
          PythonCaster<universal::MetricInvalidReason>::to_python(
              metric.reason))) {
    return nullptr;
  }
  return dict.release();
}

inline PyObject* residue_metadata_to_python(
    const universal::ResidueMetadataView& residue) {
  DictBuilder dict;
  if (!dict) {
    return nullptr;
  }
  if (!dict.set_new_ref("residue_code",
                        PyUnicode_FromStringAndSize(&residue.residue_code, 1)) ||
      !dict.set_new_ref(
          "original_residue_name",
          string_view_to_python(residue.original_residue_name)) ||
      !dict.set_new_ref("chain_id", string_view_to_python(residue.chain_id)) ||
      !dict.set_new_ref("model_id", string_view_to_python(residue.model_id)) ||
      !dict.set_new_ref("model_index",
                        PythonCaster<std::int32_t>::to_python(
                            residue.model_index)) ||
      !dict.set_new_ref("residue_number",
                        PythonCaster<std::int32_t>::to_python(
                            residue.residue_number)) ||
      !dict.set_new_ref(
          "insertion_code",
          PyUnicode_FromStringAndSize(&residue.insertion_code, 1)) ||
      !dict.set_new_ref("source_id", string_view_to_python(residue.source_id)) ||
      !dict.set_new_ref("source_filename",
                        string_view_to_python(residue.source_filename)) ||
      !dict.set_new_ref("source_residue_index",
                        PythonCaster<std::int64_t>::to_python(
                            residue.source_residue_index)) ||
      !dict.set_new_ref("source_record_index",
                        PythonCaster<std::int64_t>::to_python(
                            residue.source_record_index))) {
    return nullptr;
  }
  return dict.release();
}

inline PyObject* warnings_to_python(
    const std::vector<universal::PackageWarning>& warnings) {
  ListBuilder list(warnings.size());
  if (!list) {
    return nullptr;
  }
  for (std::size_t index = 0; index < warnings.size(); ++index) {
    if (!list.set_new_ref(
            index,
            PythonCaster<universal::PackageWarning>::to_python(
                warnings[index]))) {
      return nullptr;
    }
  }
  return list.release();
}

}  // namespace detail

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_CASTERS_HPP
