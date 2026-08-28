#ifndef HIKOBOSHI_BINDINGS_ENGINE_HELPERS_HPP
#define HIKOBOSHI_BINDINGS_ENGINE_HELPERS_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/bindings/buffer_view.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>
#include <hikoboshi/io/structure_loader.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace hikoboshi::bindings {

int raise_if_error(const universal::Status& status);

inline universal::Status invalid_argument(const char* detail) noexcept {
  return {universal::StatusCode::InvalidArgument, detail};
}

inline universal::Status failed_precondition(const char* detail) noexcept {
  return {universal::StatusCode::FailedPrecondition, detail};
}

inline bool status_ok(const universal::Status& status) noexcept {
  return status.code == universal::StatusCode::Ok;
}

inline int raise_current_error_as_invalid_argument(const char* fallback) {
  std::string detail = fallback == nullptr ? "invalid argument" : fallback;
  if (PyErr_Occurred()) {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyObjectRef type_ref(type);
    PyObjectRef value_ref(value);
    PyObjectRef traceback_ref(traceback);
    (void)type_ref;
    (void)value_ref;
    (void)traceback_ref;
    if (value != nullptr) {
      PyObjectRef text(PyObject_Str(value));
      if (text) {
        Py_ssize_t size = 0;
        const char* chars = PyUnicode_AsUTF8AndSize(text.get(), &size);
        if (chars != nullptr) {
          detail.assign(chars, static_cast<std::size_t>(size));
        } else {
          PyErr_Clear();
        }
      } else {
        PyErr_Clear();
      }
    }
  }
  return raise_if_error(invalid_argument(detail.c_str()));
}

inline universal::Result<weights::PackageHandle> package_from_python(
    PyObject* object) {
  if (object == nullptr || object == Py_None) {
    return weights::default_package(weights::kDefaultMpnnD64ModelName);
  }
  if (!PyUnicode_Check(object)) {
    return {invalid_argument("package must be a string or None"),
            {nullptr, nullptr}};
  }
  Py_ssize_t size = 0;
  const char* text = PyUnicode_AsUTF8AndSize(object, &size);
  if (text == nullptr) {
    return {{universal::StatusCode::InternalError,
             "package string conversion failed"},
            {nullptr, nullptr}};
  }
  return weights::default_package({text, static_cast<std::size_t>(size)});
}

inline universal::Result<api::Engine> make_package_engine(
    PyObject* package_object,
    std::uint32_t thread_count = 0) {
  const universal::Result<weights::PackageHandle> package =
      package_from_python(package_object);
  if (!status_ok(package.status)) {
    return {package.status, api::Engine{}};
  }
  if (package.value.descriptor == nullptr) {
    return {failed_precondition(
                "default hikoboshi-mpnn-d64 package descriptor is missing"),
            api::Engine{}};
  }

  api::EngineConfig config{};
  config.package = package.value;
  config.weights = package.value.descriptor->compatibility_views.weights;
  config.execution.thread_count = thread_count;
  return {{universal::StatusCode::Ok, ""}, api::Engine{config}};
}

inline bool parse_thread_count_arg(PyObject* object,
                                   std::uint32_t& thread_count) {
  thread_count = 0;
  if (object == nullptr) {
    return true;
  }
  if (!PyLong_Check(object) || PyBool_Check(object)) {
    raise_if_error(
        invalid_argument("thread_count must be a non-negative integer"));
    return false;
  }
  const long long value = PyLong_AsLongLong(object);
  if (value == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    raise_if_error(
        invalid_argument("thread_count is outside the supported range"));
    return false;
  }
  if (value < 0) {
    raise_if_error(
        invalid_argument("thread_count must be a non-negative integer"));
    return false;
  }
  if (static_cast<unsigned long long>(value) >
      static_cast<unsigned long long>(
          std::numeric_limits<std::uint32_t>::max())) {
    raise_if_error(
        invalid_argument("thread_count is outside the supported range"));
    return false;
  }
  thread_count = static_cast<std::uint32_t>(value);
  return true;
}

class EmbeddingInput final {
 public:
  bool acquire(PyObject* object,
               PyObject* metadata,
               const char* label,
               bool sequence_residue_codes = true) {
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
    if (!metadata_residue_codes(metadata,
                                static_cast<std::size_t>(buffer_.extent(0)),
                                residue_codes_, sequence_residue_codes)) {
      raise_current_error_as_invalid_argument(
          "metadata residue_codes are invalid");
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
            {nullptr, 0}};
  }

 private:
  static bool metadata_residue_codes(PyObject* metadata,
                                     std::size_t residue_count,
                                     std::vector<char>& out,
                                     bool sequence_residue_codes) {
    out.clear();
    if (metadata == nullptr || metadata == Py_None) {
      return true;
    }
    if (!PyDict_Check(metadata)) {
      PyErr_SetString(PyExc_TypeError, "metadata must be a dict or None");
      return false;
    }
    PyObject* codes = PyDict_GetItemString(metadata, "residue_codes");
    if (codes == nullptr || codes == Py_None) {
      return true;
    }

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
      out.assign(text, text + size);
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
      out.assign(text, text + size);
      return true;
    }

    if (!sequence_residue_codes) {
      PyErr_SetString(PyExc_TypeError,
                      "metadata residue_codes must be a string for all-vs-all");
      return false;
    }

    PyObjectRef sequence(
        PySequence_Fast(codes, "residue_codes must be a sequence"));
    if (!sequence) {
      return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.get());
    if (static_cast<std::size_t>(size) != residue_count) {
      PyErr_SetString(PyExc_ValueError,
                      "metadata residue_codes length must match embeddings");
      return false;
    }
    out.reserve(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(sequence.get());
    for (Py_ssize_t index = 0; index < size; ++index) {
      PyObject* item = items[index];
      if (!PyUnicode_Check(item)) {
        PyErr_SetString(PyExc_TypeError,
                        "each residue code must be a one-character string");
        return false;
      }
      Py_ssize_t char_size = 0;
      const char* text = PyUnicode_AsUTF8AndSize(item, &char_size);
      if (text == nullptr) {
        return false;
      }
      if (char_size < 1) {
        PyErr_SetString(PyExc_ValueError, "residue code must not be empty");
        return false;
      }
      out.push_back(text[0]);
    }
    return true;
  }

  BufferView<const float, 2> buffer_;
  std::vector<char> residue_codes_;
};

inline bool path_from_python(PyObject* object, std::string& path) {
  PyObjectRef fs_path(PyOS_FSPath(object));
  if (!fs_path) {
    return false;
  }

  if (PyUnicode_Check(fs_path.get())) {
    Py_ssize_t size = 0;
    const char* text = PyUnicode_AsUTF8AndSize(fs_path.get(), &size);
    if (text == nullptr) {
      return false;
    }
    path.assign(text, static_cast<std::size_t>(size));
    return true;
  }

  if (PyBytes_Check(fs_path.get())) {
    char* text = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(fs_path.get(), &text, &size) < 0) {
      return false;
    }
    path.assign(text, static_cast<std::size_t>(size));
    return true;
  }

  PyErr_SetString(PyExc_TypeError, "structure input must be a filesystem path");
  return false;
}

inline int apply_structure_options(PyObject* chain_index,
                                   const std::string& chain_id,
                                   const std::string& pdb_model_id,
                                   PyObject* pdb_model_index,
                                   io::StructureLoadOptions& options) {
  if (!chain_id.empty()) {
    options.chain_id = chain_id;
  }
  if (chain_index != nullptr && chain_index != Py_None) {
    if (!PyLong_Check(chain_index)) {
      return raise_if_error(invalid_argument("chain_index must be an int"));
    }
    options.chain_index = static_cast<std::size_t>(PyLong_AsSize_t(chain_index));
    if (PyErr_Occurred()) {
      return -1;
    }
  }
  if (!pdb_model_id.empty()) {
    options.model_id = pdb_model_id;
  }
  if (pdb_model_index != nullptr && pdb_model_index != Py_None) {
    if (!PyLong_Check(pdb_model_index)) {
      return raise_if_error(invalid_argument("pdb_model_index must be an int"));
    }
    options.model_index = static_cast<int>(PyLong_AsLong(pdb_model_index));
    if (PyErr_Occurred()) {
      return -1;
    }
  }
  return 0;
}

inline bool load_structure_arg(PyObject* object,
                               const io::StructureLoadOptions& options,
                               io::LoadedStructure& out) {
  std::string path;
  if (!path_from_python(object, path)) {
    return false;
  }
  universal::Result<io::LoadedStructure> loaded =
      io::load_structure_from_file(path, options);
  if (raise_if_error(loaded.status) < 0) {
    return false;
  }
  out = std::move(loaded.value);
  return true;
}

inline api::CoordsInputView coords_input_from_structure(
    const universal::StructureView& view) noexcept {
  return {view.residue_count, view.coordinates, view.atom_sources,
          view.residue_codes, view.residues};
}

inline bool optional_string_arg(PyObject* object,
                                const char* label,
                                std::string& out) {
  out.clear();
  if (object == nullptr || object == Py_None) {
    return true;
  }
  if (!PyUnicode_Check(object)) {
    PyErr_Format(PyExc_TypeError, "%s must be a string or None", label);
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

}  // namespace hikoboshi::bindings

#endif  // HIKOBOSHI_BINDINGS_ENGINE_HELPERS_HPP
