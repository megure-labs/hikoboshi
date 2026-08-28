#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/bindings/interop.hpp>
#include <hikoboshi/universal/span.hpp>

#include <cstdio>
#include <cstddef>
#include <string_view>
#include <utility>

namespace hiko_b = hikoboshi::bindings;
namespace hiko_u = hikoboshi::universal;

namespace {

struct CountingBuffer {
  PyObject_HEAD
  int getbuffer_count;
  int release_count;
  Py_ssize_t shape[2];
  Py_ssize_t strides[2];
  float values[4];
};

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

PyObject* counting_buffer_new(PyTypeObject* type, PyObject*, PyObject*) {
  auto* self = reinterpret_cast<CountingBuffer*>(type->tp_alloc(type, 0));
  if (self == nullptr) {
    return nullptr;
  }
  self->getbuffer_count = 0;
  self->release_count = 0;
  self->shape[0] = 2;
  self->shape[1] = 2;
  self->strides[0] = static_cast<Py_ssize_t>(2 * sizeof(float));
  self->strides[1] = static_cast<Py_ssize_t>(sizeof(float));
  self->values[0] = 1.0F;
  self->values[1] = 2.0F;
  self->values[2] = 3.0F;
  self->values[3] = 4.0F;
  return reinterpret_cast<PyObject*>(self);
}

int counting_buffer_getbuffer(PyObject* object, Py_buffer* view, int flags) {
  if (view == nullptr) {
    PyErr_SetString(PyExc_BufferError, "missing Py_buffer");
    return -1;
  }
  auto* self = reinterpret_cast<CountingBuffer*>(object);
  ++self->getbuffer_count;

  view->buf = self->values;
  view->obj = object;
  Py_INCREF(object);
  view->len = static_cast<Py_ssize_t>(sizeof(self->values));
  view->readonly = 0;
  view->itemsize = static_cast<Py_ssize_t>(sizeof(float));
  view->format = (flags & PyBUF_FORMAT) != 0 ? const_cast<char*>("f") : nullptr;
  view->ndim = 2;
  view->shape = (flags & PyBUF_ND) != 0 ? self->shape : nullptr;
  view->strides = (flags & PyBUF_STRIDES) != 0 ? self->strides : nullptr;
  view->suboffsets = nullptr;
  view->internal = nullptr;
  return 0;
}

void counting_buffer_releasebuffer(PyObject* object, Py_buffer*) {
  auto* self = reinterpret_cast<CountingBuffer*>(object);
  ++self->release_count;
}

hiko_b::PyObjectRef make_counting_buffer_type() {
  static PyType_Slot slots[] = {
      {Py_tp_new, reinterpret_cast<void*>(counting_buffer_new)},
      {Py_bf_getbuffer, reinterpret_cast<void*>(counting_buffer_getbuffer)},
      {Py_bf_releasebuffer,
       reinterpret_cast<void*>(counting_buffer_releasebuffer)},
      {0, nullptr},
  };
  static PyType_Spec spec = {
      "hikoboshi_test.CountingBuffer",
      sizeof(CountingBuffer),
      0,
      Py_TPFLAGS_DEFAULT,
      slots,
  };
  return hiko_b::PyObjectRef::adopt(PyType_FromSpec(&spec));
}

bool test_pyobject_ref() {
  hiko_b::PyObjectRef list(PyList_New(0));
  if (!require(static_cast<bool>(list), "failed to create list")) {
    return false;
  }

  PyObject* raw = list.get();
  const Py_ssize_t base_refcnt = Py_REFCNT(raw);
  {
    hiko_b::PyObjectRef borrowed = hiko_b::PyObjectRef::borrow(raw);
    if (!require(Py_REFCNT(raw) == base_refcnt + 1,
                 "borrowed ref did not incref")) {
      return false;
    }
    hiko_b::PyObjectRef moved(std::move(borrowed));
    if (!require(!borrowed && moved.get() == raw,
                 "move did not transfer ownership")) {
      return false;
    }
  }

  if (!require(Py_REFCNT(raw) == base_refcnt,
               "borrowed ref did not decref")) {
    return false;
  }
  PyObject* released = list.release();
  if (!require(!list && released == raw, "release did not return owned ref")) {
    Py_XDECREF(released);
    return false;
  }
  Py_DECREF(released);
  return true;
}

bool test_builders_and_span_conversion() {
  hiko_b::DictBuilder dict;
  if (!require(dict.ok(), "failed to create dict builder")) {
    return false;
  }
  if (!require(dict.set_long("answer", 42), "failed to set long") ||
      !require(dict.set_bool("ok", true), "failed to set bool") ||
      !require(dict.set_string_view("name", std::string_view{"hikoboshi"}),
               "failed to set string view")) {
    return false;
  }

  hiko_b::PyObjectRef built_dict(dict.release());
  PyObject* answer = PyDict_GetItemString(built_dict.get(), "answer");
  if (!require(answer != nullptr && PyLong_AsLong(answer) == 42,
               "dict builder stored the wrong value")) {
    return false;
  }

  hiko_b::ListBuilder list(1);
  if (!require(list.ok(), "failed to create list builder") ||
      !require(list.set_new_ref(0, PyUnicode_FromString("first")),
               "failed to set list item")) {
    return false;
  }
  if (!require(!list.set_new_ref(2, PyLong_FromLong(2)),
               "out-of-range list set unexpectedly succeeded")) {
    return false;
  }
  if (!require(PyErr_ExceptionMatches(PyExc_IndexError),
               "out-of-range list set raised the wrong error")) {
    return false;
  }
  PyErr_Clear();

  const int values[3] = {1, 2, 3};
  const hiko_u::Span<const int> value_span{values, 3};
  hiko_b::PyObjectRef py_values(hiko_b::span_to_pylist(value_span));
  if (!require(static_cast<bool>(py_values) && PyList_Check(py_values.get()),
               "span_to_pylist did not create a list")) {
    return false;
  }
  if (!require(PyList_GET_SIZE(py_values.get()) == 3,
               "span_to_pylist produced wrong length")) {
    return false;
  }
  PyObject* second = PyList_GET_ITEM(py_values.get(), 1);
  if (!require(second != nullptr && PyLong_AsLong(second) == 2,
               "span_to_pylist produced wrong scalar value")) {
    return false;
  }

  const char residues[2] = {'A', 'G'};
  const hiko_u::Span<const char> residue_span{residues, 2};
  hiko_b::PyObjectRef py_residues(hiko_b::span_to_pylist(residue_span));
  PyObject* residue = py_residues ? PyList_GET_ITEM(py_residues.get(), 0) : nullptr;
  if (!require(residue != nullptr && PyUnicode_Check(residue),
               "char span did not convert to unicode items")) {
    return false;
  }
  return true;
}

bool test_buffer_view_release() {
  hiko_b::PyObjectRef type = make_counting_buffer_type();
  if (!require(static_cast<bool>(type), "failed to create counting buffer type")) {
    return false;
  }
  hiko_b::PyObjectRef object(PyObject_CallNoArgs(type.get()));
  if (!require(static_cast<bool>(object), "failed to create counting buffer")) {
    return false;
  }
  auto* exporter = reinterpret_cast<CountingBuffer*>(object.get());

  {
    hiko_b::BufferView<const float, 2> view;
    if (!require(view.acquire(object.get(), "embeddings", hiko_b::BufferContiguity::C),
                 "failed to acquire valid buffer")) {
      return false;
    }
    if (!require(view.active(), "valid buffer view is not active") ||
        !require(exporter->getbuffer_count == 1, "getbuffer was not called") ||
        !require(exporter->release_count == 0, "buffer released too early") ||
        !require(view.extent(0) == 2 && view.extent(1) == 2,
                 "buffer shape mismatch") ||
        !require(view.stride(0) == static_cast<Py_ssize_t>(2 * sizeof(float)),
                 "buffer stride mismatch") ||
        !require(view.element_count() == 4, "buffer element count mismatch") ||
        !require(view.data()[3] == 4.0F, "buffer data mismatch")) {
      return false;
    }
  }

  return require(exporter->release_count == 1, "buffer release was not called");
}

bool test_error_paths() {
  hiko_b::PyObjectRef bytes(PyBytes_FromStringAndSize("abcd", 4));
  if (!require(static_cast<bool>(bytes), "failed to create bytes object")) {
    return false;
  }

  hiko_b::BufferView<const float, 2> view;
  if (!require(!view.acquire(bytes.get(), "bad buffer", hiko_b::BufferContiguity::C),
               "invalid buffer unexpectedly acquired")) {
    return false;
  }
  if (!require(!view.active(), "failed buffer acquire stayed active") ||
      !require(PyErr_Occurred() != nullptr, "failed buffer acquire set no error")) {
    return false;
  }
  PyErr_Clear();

  hiko_b::ListBuilder list(0);
  if (!require(!list.set_new_ref(1, PyLong_FromLong(7)),
               "invalid list set unexpectedly succeeded")) {
    return false;
  }
  if (!require(PyErr_ExceptionMatches(PyExc_IndexError),
               "invalid list set raised the wrong error")) {
    return false;
  }
  PyErr_Clear();
  return true;
}

}  // namespace

int main() {
  Py_Initialize();

  const bool ok = test_pyobject_ref() && test_builders_and_span_conversion() &&
                  test_buffer_view_release() && test_error_paths();

  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  const int finalize_status = Py_FinalizeEx();
  return ok && finalize_status == 0 ? 0 : 1;
}
