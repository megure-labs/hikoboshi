#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <hikoboshi/bindings/args.hpp>
#include <hikoboshi/bindings/dict_builder.hpp>
#include <hikoboshi/bindings/py_object_ref.hpp>

#include <cmath>
#include <cstdio>
#include <string>

namespace hiko_b = hikoboshi::bindings;

namespace {

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

bool test_required_and_optional_keywords() {
  hiko_b::DictBuilder kwargs;
  hiko_b::PyObjectRef package(PyUnicode_FromString("mpnn64"));
  if (!require(kwargs.ok(), "failed to create kwargs") ||
      !require(kwargs.set_string_view("path", "query.pdb"),
               "failed to set path kwarg") ||
      !require(kwargs.set_double("gap_open", -1.25),
               "failed to set gap_open kwarg") ||
      !require(kwargs.set_bool("include_self", true),
               "failed to set include_self kwarg") ||
      !require(kwargs.set_borrowed("package", package.get()),
               "failed to set package kwarg")) {
    return false;
  }
  hiko_b::PyObjectRef args(PyTuple_New(0));

  std::string path;
  double gap_open = 0.0;
  bool include_self = false;
  PyObject* parsed_package = nullptr;
  hiko_b::TypedArgParser parser(args.get(), kwargs.get(), "pairwise");
  if (!require(parser.required("path", path), "failed to parse required path") ||
      !require(parser.optional("gap_open", gap_open, -1.4),
               "failed to parse optional gap_open") ||
      !require(parser.optional("include_self", include_self, false),
               "failed to parse optional include_self") ||
      !require(parser.optional_object("package", parsed_package, Py_None),
               "failed to parse optional package") ||
      !require(parser.finish(), "parser finish failed")) {
    return false;
  }

  return require(path == "query.pdb", "parsed path mismatch") &&
         require(std::fabs(gap_open + 1.25) < 1.0e-9,
                 "parsed gap_open mismatch") &&
         require(include_self, "parsed bool mismatch") &&
         require(parsed_package == package.get(), "parsed package mismatch");
}

bool test_positional_and_default_values() {
  hiko_b::PyObjectRef path(PyUnicode_FromString("target.pdb"));
  hiko_b::PyObjectRef args(PyTuple_Pack(1, path.get()));
  hiko_b::PyObjectRef kwargs(PyDict_New());

  std::string parsed_path;
  double gap_extension = 0.0;
  hiko_b::TypedArgParser parser(args.get(), kwargs.get(), "pairwise");
  if (!require(parser.required("path", parsed_path),
               "failed to parse positional path") ||
      !require(parser.optional("gap_extension", gap_extension, -0.15),
               "failed to parse defaulted optional") ||
      !require(parser.finish(), "parser finish failed for positional args")) {
    return false;
  }
  return require(parsed_path == "target.pdb", "positional path mismatch") &&
         require(std::fabs(gap_extension + 0.15) < 1.0e-9,
                 "default optional value mismatch");
}

bool expect_type_error(bool value, const char* message) {
  if (!require(!value, message)) {
    return false;
  }
  if (!require(PyErr_ExceptionMatches(PyExc_TypeError),
               "parser failure did not raise TypeError")) {
    return false;
  }
  PyErr_Clear();
  return true;
}

bool test_missing_required_failure() {
  hiko_b::PyObjectRef args(PyTuple_New(0));
  hiko_b::PyObjectRef kwargs(PyDict_New());
  std::string path;
  hiko_b::TypedArgParser parser(args.get(), kwargs.get(), "encode");
  return expect_type_error(parser.required("path", path),
                           "missing required argument unexpectedly parsed");
}

bool test_unknown_keyword_failure() {
  hiko_b::DictBuilder kwargs;
  if (!require(kwargs.ok(), "failed to create kwargs") ||
      !require(kwargs.set_long("unexpected", 7),
               "failed to set unexpected kwarg")) {
    return false;
  }
  hiko_b::PyObjectRef args(PyTuple_New(0));
  double gap_open = 0.0;
  hiko_b::TypedArgParser parser(args.get(), kwargs.get(), "pairwise");
  if (!require(parser.optional("gap_open", gap_open, -1.4),
               "known optional failed before unknown keyword check")) {
    return false;
  }
  return expect_type_error(parser.finish(),
                           "unknown keyword unexpectedly accepted");
}

bool test_duplicate_argument_failure() {
  hiko_b::PyObjectRef positional(PyUnicode_FromString("positional.pdb"));
  hiko_b::PyObjectRef args(PyTuple_Pack(1, positional.get()));
  hiko_b::DictBuilder kwargs;
  if (!require(kwargs.ok(), "failed to create kwargs") ||
      !require(kwargs.set_string_view("path", "keyword.pdb"),
               "failed to set duplicate path kwarg")) {
    return false;
  }

  std::string path;
  hiko_b::TypedArgParser parser(args.get(), kwargs.get(), "encode");
  return expect_type_error(parser.required("path", path),
                           "duplicate argument unexpectedly accepted");
}

bool test_bad_type_failure() {
  hiko_b::DictBuilder kwargs;
  if (!require(kwargs.ok(), "failed to create kwargs") ||
      !require(kwargs.set_string_view("gap_open", "wide"),
               "failed to set bad gap_open kwarg")) {
    return false;
  }
  hiko_b::PyObjectRef args(PyTuple_New(0));

  double gap_open = 0.0;
  hiko_b::TypedArgParser parser(args.get(), kwargs.get(), "pairwise");
  return expect_type_error(
      parser.optional("gap_open", gap_open, -1.4),
      "string unexpectedly parsed as floating-point kwarg");
}

}  // namespace

int main() {
  Py_Initialize();

  const bool ok = test_required_and_optional_keywords() &&
                  test_positional_and_default_values() &&
                  test_missing_required_failure() &&
                  test_unknown_keyword_failure() &&
                  test_duplicate_argument_failure() &&
                  test_bad_type_failure();

  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  const int finalize_status = Py_FinalizeEx();
  return ok && finalize_status == 0 ? 0 : 1;
}
