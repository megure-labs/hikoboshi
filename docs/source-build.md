# Source Builds

Use a source build when you need the standalone native CLI and C++ SDK, an
editable Python install, or direct access to a build tree. Hikoboshi `0.1.1`
uses Meson as its native build graph and `meson-python` as its Python build
backend.

PyPI and Conda packages are coming soon; the commands below install from a
source checkout.

## Prerequisites

- Python `3.10+` to run Meson and the deterministic model-blob generator
- Meson `1.1+`
- A C++17 compiler
- `meson-python>=0.15` for Python package builds
- Optional: NumPy `1.24+` for NumPy array conversion in the Python adapter

Hikoboshi `0.1.1` is a scalar CPU build. Reserved GPU and SIMD backend options
remain disabled by default and are not buildable release backends in this
version.

## Native CLI and C++ SDK

```bash
meson setup builddir -Dhikoboshi_python_api=false
meson compile -C builddir -j 8
meson test -C builddir
```

This is the default profile. It does not discover Python development headers,
build the CPython extension, install the Python package, or link the CLI to
CPython. Python is used only as a build tool by Meson and the checked artifact
generator; the resulting `hikoboshi` executable has no Python runtime
dependency.

The build tree provides the native CLI at:

```bash
./builddir/hikoboshi
```

Run basic diagnostics from the build tree:

```bash
./builddir/hikoboshi version
./builddir/hikoboshi info
./builddir/hikoboshi info backends
./builddir/hikoboshi info models
```

## Python Install

From a source checkout:

```bash
python -m pip install .
```

`meson-python` selects `-Dhikoboshi_python_api=true` for package builds. To
build and test the Python API directly with Meson instead:

```bash
meson setup build-python -Dhikoboshi_python_api=true
meson compile -C build-python -j 8
meson test -C build-python --print-errorlogs
```

With optional NumPy adapter support:

```bash
python -m pip install '.[numpy]'
```

Editable installs use the same extras:

```bash
python -m pip install -e .
python -m pip install -e '.[numpy]'
```

## Running Python From A Build Tree

For local smoke tests without installing into the environment, point Python at
the source package and compiled extension:

```bash
HIKOBOSHI_BUILD_ROOT="$PWD/build-python" PYTHONPATH="$PWD/python" python - <<'PY'
import hikoboshi as hkbs

print(hkbs.__version__)
print(hkbs.version_info())
PY
```

## Backend Options

The default Meson options build the scalar backend:

```bash
meson setup builddir -Dhikoboshi_python_api=false
```

Hikoboshi `0.1.1` accepts `auto` and `scalar` at runtime. Build options for CUDA,
HIP, Metal, Vulkan, OpenCL, and CPU SIMD names are reserved diagnostics in this
version; enabling them causes Meson configuration to fail with an explanatory
message.

## Cleaning

Remove the build directory to force a fresh native build:

```bash
rm -rf builddir
meson setup builddir -Dhikoboshi_python_api=false
meson compile -C builddir -j 8
```

## CPU targeting

Distributable builds default to `-Dhikoboshi_cpu_target=portable`. This adds no
host-specific compiler flags and preserves the toolchain's baseline target.
Explicit `cpp_args` or cross-file CPU flags still apply; portable mode does not
remove flags supplied by the builder.

For a binary used on the build machine (or compatible CPU hardware), opt into
native targeting:

```bash
meson setup build-native --buildtype=release -Dhikoboshi_cpu_target=native
meson compile -C build-native -j 8
meson test -C build-native --print-errorlogs
```

Native mode uses GCC/Clang `-march=native` on x86 or `-mcpu=native` on ARM.
It keeps the scalar backend and existing fast/strict GEMM selection. Native
mode also sets `-ffp-contract=off` to preserve separate multiply/add rounding
when the compiler enables FMA instructions. It does not add fast-math or
hand-written SIMD kernels. Meson reports the selected
mode and flags. Cross builds reject native mode because the build CPU may differ
from the deployment CPU; use portable plus an explicit cross toolchain instead.
Unsupported compilers/CPU families fail clearly when native is requested.

A native binary may use instructions absent on another machine. It has no
runtime instruction-set fallback: retain a separate portable build for broader
deployment, or rebuild with the portable default on the destination toolchain.
Do not distribute a native-built wheel as a generic portable wheel.

```bash
meson setup build-portable --buildtype=release -Dhikoboshi_cpu_target=portable
meson compile -C build-portable -j 8
```

Benchmark portable and native builds on the same CPU and inputs, and check
numerical parity in the selected GEMM mode before adopting the native build.
