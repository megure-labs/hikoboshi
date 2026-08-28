project(
  'hikoboshi',
  'cpp',
  version: '0.1.0',
  meson_version: '>=1.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
  ],
)

cxx = meson.get_compiler('cpp')
summary('C++ compiler', cxx.get_id() + ' ' + cxx.version(), section: 'Hikoboshi toolchain')

hikoboshi_build_sanitize = get_option('b_sanitize')
hikoboshi_build_werror = get_option('werror')
hikoboshi_bench_no_deterministic_sink = get_option(
  'hikoboshi_bench_no_deterministic_sink',
)
hikoboshi_bench_mpnn_dump = get_option('hikoboshi_bench_mpnn_dump')
hikoboshi_allvsall_embedding_storage = get_option(
  'hikoboshi_allvsall_embedding_storage',
)
hikoboshi_allvsall_sink = get_option('hikoboshi_allvsall_sink')
hikoboshi_allvsall_pair_scheduler = get_option('hikoboshi_allvsall_pair_scheduler')
hikoboshi_lto_option = get_option('hikoboshi_lto')
hikoboshi_b_lto = get_option('b_lto')
hikoboshi_lto_effective = hikoboshi_lto_option or hikoboshi_b_lto
hikoboshi_gemm_parity_mode = get_option('hikoboshi_gemm_parity_mode')
hikoboshi_python_api = get_option('hikoboshi_python_api')
hikoboshi_tsan_executable_link_args = []
hikoboshi_tsan_setarch = disabler()
hikoboshi_tsan_setarch_arch = ''
hikoboshi_tsan_setarch_found = false
hikoboshi_gnu_ld_wrap_supported = host_machine.system() == 'linux'
hikoboshi_mpnn_forward_wrap_link_args = []
if hikoboshi_gnu_ld_wrap_supported
  hikoboshi_mpnn_forward_wrap_link_args += ['-Wl,--wrap=_ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE']
endif
if hikoboshi_build_sanitize == 'thread' and host_machine.system() == 'linux'
  hikoboshi_tsan_executable_link_args += ['-no-pie']
  hikoboshi_tsan_setarch = find_program('setarch', required: false)
  hikoboshi_tsan_setarch_arch = host_machine.cpu_family()
  hikoboshi_tsan_setarch_found = hikoboshi_tsan_setarch.found()
endif

summary(
  {
    'b_sanitize': hikoboshi_build_sanitize,
    'werror': hikoboshi_build_werror,
    'hikoboshi_bench_no_deterministic_sink': hikoboshi_bench_no_deterministic_sink,
    'hikoboshi_bench_mpnn_dump': hikoboshi_bench_mpnn_dump,
    'hikoboshi_allvsall_embedding_storage': hikoboshi_allvsall_embedding_storage,
    'hikoboshi_allvsall_sink': hikoboshi_allvsall_sink,
    'hikoboshi_allvsall_pair_scheduler': hikoboshi_allvsall_pair_scheduler,
    'hikoboshi_lto': hikoboshi_lto_option,
    'b_lto': hikoboshi_b_lto,
    'lto effective': hikoboshi_lto_effective,
    'hikoboshi_gemm_parity_mode': hikoboshi_gemm_parity_mode,
    'hikoboshi_python_api': hikoboshi_python_api,
    'tsan executable link args': hikoboshi_tsan_executable_link_args,
    'tsan setarch wrapper': hikoboshi_tsan_setarch_found,
  },
  section: 'Hikoboshi build options',
)

if hikoboshi_lto_option and not hikoboshi_b_lto
  add_project_arguments('-flto', language: 'cpp')
  add_project_link_arguments('-flto', language: 'cpp')
endif

if hikoboshi_bench_no_deterministic_sink
  add_project_arguments(
    '-DHIKOBOSHI_BENCH_NO_DETERMINISTIC_SINK=1',
    language: 'cpp',
  )
endif

if hikoboshi_bench_mpnn_dump
  add_project_arguments(
    '-DHIKOBOSHI_BENCH_MPNN_DUMP=1',
    language: 'cpp',
  )
endif

if hikoboshi_allvsall_embedding_storage == 'bucketed'
  add_project_arguments(
    '-DHIKOBOSHI_ALLVSALL_EMBEDDING_STORAGE_BUCKETED=1',
    language: 'cpp',
  )
endif

if hikoboshi_allvsall_sink == 'streaming'
  if hikoboshi_bench_no_deterministic_sink
    error(
      'hikoboshi_allvsall_sink=streaming and hikoboshi_bench_no_deterministic_sink=true are mutually exclusive; the bench-no-deterministic-sink path is a separate measurement hook and must be combined with hikoboshi_allvsall_sink=buffered.',
    )
  endif
  add_project_arguments(
    '-DHIKOBOSHI_ALLVSALL_SINK_STREAMING=1',
    language: 'cpp',
  )
endif

if hikoboshi_allvsall_pair_scheduler == 'query_tiled'
  add_project_arguments(
    '-DHIKOBOSHI_ALLVSALL_PAIR_SCHEDULER_QUERY_TILED=1',
    language: 'cpp',
  )
endif

if hikoboshi_gemm_parity_mode == 'fast'
  add_project_arguments(
    '-DHIKOBOSHI_GEMM_PARITY_MODE_DEFAULT_FAST=1',
    language: 'cpp',
  )
endif

hikoboshi_backend_cuda = get_option('hikoboshi_backend_cuda')
hikoboshi_backend_hip = get_option('hikoboshi_backend_hip')
hikoboshi_backend_metal = get_option('hikoboshi_backend_metal')
hikoboshi_backend_vulkan = get_option('hikoboshi_backend_vulkan')
hikoboshi_backend_opencl = get_option('hikoboshi_backend_opencl')
hikoboshi_backend_simd = get_option('hikoboshi_backend_simd')
hikoboshi_backend_cuda_archs = get_option('hikoboshi_backend_cuda_archs')
hikoboshi_backend_hip_archs = get_option('hikoboshi_backend_hip_archs')

hikoboshi_backend_cuda_state = 'disabled'
if hikoboshi_backend_cuda.enabled()
  hikoboshi_backend_cuda_state = 'enabled'
elif hikoboshi_backend_cuda.auto()
  hikoboshi_backend_cuda_state = 'auto'
endif

hikoboshi_backend_hip_state = 'disabled'
if hikoboshi_backend_hip.enabled()
  hikoboshi_backend_hip_state = 'enabled'
elif hikoboshi_backend_hip.auto()
  hikoboshi_backend_hip_state = 'auto'
endif

hikoboshi_backend_metal_state = 'disabled'
if hikoboshi_backend_metal.enabled()
  hikoboshi_backend_metal_state = 'enabled'
elif hikoboshi_backend_metal.auto()
  hikoboshi_backend_metal_state = 'auto'
endif

hikoboshi_backend_vulkan_state = 'disabled'
if hikoboshi_backend_vulkan.enabled()
  hikoboshi_backend_vulkan_state = 'enabled'
elif hikoboshi_backend_vulkan.auto()
  hikoboshi_backend_vulkan_state = 'auto'
endif

hikoboshi_backend_opencl_state = 'disabled'
if hikoboshi_backend_opencl.enabled()
  hikoboshi_backend_opencl_state = 'enabled'
elif hikoboshi_backend_opencl.auto()
  hikoboshi_backend_opencl_state = 'auto'
endif

summary(
  {
    'compiled backend': 'scalar',
    'hikoboshi_backend_cuda': hikoboshi_backend_cuda_state,
    'hikoboshi_backend_hip': hikoboshi_backend_hip_state,
    'hikoboshi_backend_metal': hikoboshi_backend_metal_state,
    'hikoboshi_backend_vulkan': hikoboshi_backend_vulkan_state,
    'hikoboshi_backend_opencl': hikoboshi_backend_opencl_state,
    'hikoboshi_backend_simd': hikoboshi_backend_simd,
    'hikoboshi_backend_cuda_archs': hikoboshi_backend_cuda_archs,
    'hikoboshi_backend_hip_archs': hikoboshi_backend_hip_archs,
  },
  section: 'Hikoboshi backend options',
)

hikoboshi_reserved_backend_requests = []
if hikoboshi_backend_cuda.enabled()
  hikoboshi_reserved_backend_requests += ['hikoboshi_backend_cuda=enabled']
endif
if hikoboshi_backend_hip.enabled()
  hikoboshi_reserved_backend_requests += ['hikoboshi_backend_hip=enabled']
endif
if hikoboshi_backend_metal.enabled()
  hikoboshi_reserved_backend_requests += ['hikoboshi_backend_metal=enabled']
endif
if hikoboshi_backend_vulkan.enabled()
  hikoboshi_reserved_backend_requests += ['hikoboshi_backend_vulkan=enabled']
endif
if hikoboshi_backend_opencl.enabled()
  hikoboshi_reserved_backend_requests += ['hikoboshi_backend_opencl=enabled']
endif
if hikoboshi_backend_simd != 'none'
  hikoboshi_reserved_backend_requests += ['hikoboshi_backend_simd=' + hikoboshi_backend_simd]
endif

if hikoboshi_reserved_backend_requests.length() > 0
  hikoboshi_reserved_backend_diagnostic = ''
  foreach request : hikoboshi_reserved_backend_requests
    if hikoboshi_reserved_backend_diagnostic == ''
      hikoboshi_reserved_backend_diagnostic = request
    else
      hikoboshi_reserved_backend_diagnostic += ', ' + request
    endif
  endforeach

  error(
    'Reserved Hikoboshi backend options are not buildable in 0.1.0 scalar-only builds: '
    + hikoboshi_reserved_backend_diagnostic
    + '. Set GPU backends to disabled and hikoboshi_backend_simd=none.',
  )
endif

python_mod = import('python')
codegen_python = python_mod.find_installation(required: true)
if hikoboshi_python_api
  python = python_mod.find_installation(required: true, pure: false)
  python_dep = python.dependency(required: true)
  python_embed_dep = python.dependency(required: true, embed: true)
endif
threads_dep = dependency('threads')
bash_program = find_program('bash', required: true)

hikoboshi_public_inc = include_directories('cpp/include')

subdir('cpp/runtime')
subdir('cpp/core')

hikoboshi_universal_headers = files(
  'cpp/include/hikoboshi/universal/alignment_path.hpp',
  'cpp/include/hikoboshi/universal/backend.hpp',
  'cpp/include/hikoboshi/universal/embedding.hpp',
  'cpp/include/hikoboshi/universal/execution_options.hpp',
  'cpp/include/hikoboshi/universal/inline.hpp',
  'cpp/include/hikoboshi/universal/metrics.hpp',
  'cpp/include/hikoboshi/universal/package.hpp',
  'cpp/include/hikoboshi/universal/planner.hpp',
  'cpp/include/hikoboshi/universal/span.hpp',
  'cpp/include/hikoboshi/universal/status.hpp',
  'cpp/include/hikoboshi/universal/structure.hpp',
  'cpp/include/hikoboshi/universal/tensor.hpp',
  'cpp/include/hikoboshi/universal/tensor_role.hpp',
  'cpp/include/hikoboshi/universal/types.hpp',
  'cpp/include/hikoboshi/universal/version.hpp',
  'cpp/include/hikoboshi/universal/weights.hpp',
)
install_headers(hikoboshi_universal_headers, subdir: 'hikoboshi/universal')

hikoboshi_api_headers = files(
  'cpp/include/hikoboshi/api/all_vs_all.hpp',
  'cpp/include/hikoboshi/api/engine.hpp',
  'cpp/include/hikoboshi/api/requests.hpp',
  'cpp/include/hikoboshi/api/results.hpp',
  'cpp/include/hikoboshi/api/score_alignment.hpp',
  'cpp/include/hikoboshi/api/types.hpp',
  'cpp/include/hikoboshi/api/version.hpp',
)
install_headers(hikoboshi_api_headers, subdir: 'hikoboshi/api')

hikoboshi_weights_headers = files(
  'cpp/include/hikoboshi/weights/manifest.hpp',
  'cpp/include/hikoboshi/weights/provider.hpp',
  'cpp/include/hikoboshi/weights/provider_register.hpp',
)
install_headers(hikoboshi_weights_headers, subdir: 'hikoboshi/weights')

subdir('cpp/universal')
subdir('cpp/weights')
subdir('cpp/primitives')
subdir('cpp/dispatch')
subdir('cpp/modules')
subdir('cpp/algorithms')
subdir('cpp/api')
subdir('cpp/io')
subdir('cpp/errors')
subdir('cpp/cli')
subdir('cpp/bindings')

hikoboshi_weights_blob_lib = static_library(
  'hikoboshi_weights_blob',
  hikoboshi_weights_blob_sources,
  cpp_args: ['-O0'],
  override_options: ['warning_level=0', 'werror=false'],
  include_directories: [
    hikoboshi_public_inc,
    include_directories('cpp/weights/generated'),
  ],
)

hikoboshi_universal_lib = static_library(
  'hikoboshi_universal',
  hikoboshi_universal_sources,
  dependencies: [threads_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_universal_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [threads_dep],
  link_with: hikoboshi_universal_lib,
)

hikoboshi_primitives_lib = static_library(
  'hikoboshi_primitives',
  hikoboshi_primitives_sources,
  dependencies: [hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_primitives_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_universal_dep],
  link_with: hikoboshi_primitives_lib,
)

hikoboshi_dispatch_lib = static_library(
  'hikoboshi_dispatch',
  hikoboshi_dispatch_sources,
  dependencies: [hikoboshi_primitives_dep, hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_dispatch_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_primitives_dep, hikoboshi_universal_dep],
  link_with: hikoboshi_dispatch_lib,
)

hikoboshi_weights_lib = static_library(
  'hikoboshi_weights',
  hikoboshi_weights_sources,
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_weights_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_universal_dep],
  link_with: [hikoboshi_weights_lib, hikoboshi_weights_blob_lib],
)

hikoboshi_modules_lib = static_library(
  'hikoboshi_modules',
  hikoboshi_modules_sources,
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_modules_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_universal_dep],
  link_with: hikoboshi_modules_lib,
)

hikoboshi_algorithms_lib = static_library(
  'hikoboshi_algorithms',
  hikoboshi_algorithms_sources,
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_algorithms_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
  link_with: hikoboshi_algorithms_lib,
)

hikoboshi_api_lib = static_library(
  'hikoboshi_api',
  hikoboshi_api_sources,
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_api_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  link_with: hikoboshi_api_lib,
)

hikoboshi_io_lib = static_library(
  'hikoboshi_io',
  hikoboshi_io_sources,
  dependencies: [hikoboshi_api_dep, hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_io_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_api_dep, hikoboshi_universal_dep],
  link_with: hikoboshi_io_lib,
)

hikoboshi_errors_lib = static_library(
  'hikoboshi_errors',
  hikoboshi_errors_sources,
  dependencies: [hikoboshi_universal_dep],
  include_directories: hikoboshi_public_inc,
)
hikoboshi_errors_dep = declare_dependency(
  include_directories: hikoboshi_public_inc,
  dependencies: [hikoboshi_universal_dep],
  link_with: hikoboshi_errors_lib,
)

hikoboshi_cli = executable(
  'hikoboshi_cli',
  hikoboshi_cli_sources,
  dependencies: [
    hikoboshi_api_dep,
    hikoboshi_io_dep,
    hikoboshi_weights_dep,
    hikoboshi_errors_dep,
  ],
)

copy_program = find_program('cp', required: true)
hikoboshi_cli_install_name = custom_target(
  'hikoboshi_cli_install_name',
  input: hikoboshi_cli,
  output: 'hikoboshi',
  command: [copy_program, '@INPUT@', '@OUTPUT@'],
  build_by_default: true,
  install: true,
  install_dir: get_option('bindir'),
)

if hikoboshi_python_api
  hikoboshi_pyext_module = python.extension_module(
    '_core',
    hikoboshi_pyext_sources,
    dependencies: [
      python_dep,
      hikoboshi_api_dep,
      hikoboshi_io_dep,
      hikoboshi_weights_dep,
    ],
    include_directories: hikoboshi_public_inc,
    install: true,
    subdir: 'hikoboshi',
  )
  hikoboshi_pyext = alias_target('hikoboshi_pyext', hikoboshi_pyext_module)
endif

hikoboshi_sdk = alias_target(
  'hikoboshi_sdk',
  hikoboshi_api_lib,
  hikoboshi_weights_lib,
  hikoboshi_weights_blob_lib,
)
hikoboshi_sdk_dep = declare_dependency(
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep],
)

if hikoboshi_python_api
  hikoboshi_python_sources = files(
    'python/hikoboshi/__init__.py',
    'python/hikoboshi/_arrays.py',
    'python/hikoboshi/_chain_resolve.py',
    'python/hikoboshi/_core.pyi',
    'python/hikoboshi/_input_detect.py',
    'python/hikoboshi/all_vs_all.py',
    'python/hikoboshi/encode.py',
    'python/hikoboshi/exceptions.py',
    'python/hikoboshi/info.py',
    'python/hikoboshi/inverse_fold.py',
    'python/hikoboshi/pair_list.py',
    'python/hikoboshi/pairwise.py',
    'python/hikoboshi/py.typed',
    'python/hikoboshi/results.py',
    'python/hikoboshi/score_alignment.py',
  )

  python.install_sources(
    hikoboshi_python_sources,
    subdir: 'hikoboshi',
  )

endif

hikoboshi_alignment_path_smoke = executable(
  'hikoboshi_alignment_path_smoke',
  'tests/cpp/alignment_path_smoke.cpp',
  dependencies: [hikoboshi_universal_dep],
)
hikoboshi_target_graph_smoke = executable(
  'hikoboshi_target_graph_smoke',
  'tests/cpp/target_graph_smoke.cpp',
  dependencies: [hikoboshi_sdk_dep],
)
hikoboshi_universal_public_headers = executable(
  'hikoboshi_universal_public_headers',
  'tests/cpp/universal_public_headers.cpp',
  dependencies: [hikoboshi_universal_dep],
)
hikoboshi_proteinmpnn_v48_020_contract_test = executable(
  'hikoboshi_proteinmpnn_v48_020_contract_test',
  'tests/unit/proteinmpnn_v48_020_contract_test.cpp',
  dependencies: [hikoboshi_universal_dep],
)
hikoboshi_proteinmpnn_decoder_layer_parity_test = executable(
  'hikoboshi_proteinmpnn_decoder_layer_parity_test',
  'tests/unit/proteinmpnn_decoder_layer_parity_test.cpp',
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_reserved_workspace_types_smoke = executable(
  'hikoboshi_reserved_workspace_types_smoke',
  'tests/cpp/reserved_workspace_types_smoke.cpp',
  dependencies: [hikoboshi_universal_dep],
)
hikoboshi_api_public_headers = executable(
  'hikoboshi_api_public_headers',
  'tests/cpp/api_public_headers.cpp',
  dependencies: [hikoboshi_api_dep],
)
hikoboshi_weights_public_headers = executable(
  'hikoboshi_weights_public_headers',
  'tests/cpp/weights_public_headers.cpp',
  dependencies: [hikoboshi_weights_dep],
)
hikoboshi_weights_package_provider_tests = executable(
  'hikoboshi_weights_package_provider_tests',
  'tests/cpp/weights_package_provider_tests.cpp',
  dependencies: [hikoboshi_weights_dep],
)
hikoboshi_tensor_slot_smoke = executable(
  'hikoboshi_tensor_slot_smoke',
  'tests/cpp/tensor_slot_smoke.cpp',
  dependencies: [hikoboshi_weights_dep],
)
hikoboshi_package_registry_tests = executable(
  'hikoboshi_package_registry_tests',
  'tests/cpp/package_registry_tests.cpp',
  dependencies: [hikoboshi_weights_dep],
)
hikoboshi_package_validation_pipeline_tests = executable(
  'hikoboshi_package_validation_pipeline_tests',
  'tests/cpp/package_validation_pipeline_tests.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_package_validation_noalloc_test = executable(
  'hikoboshi_package_validation_noalloc_test',
  'tests/cpp/package_validation_noalloc_test.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_core_dispatch_seam_test = executable(
  'test_dispatch_seam',
  'tests/core/test_dispatch_seam.cpp',
  dependencies: [hikoboshi_core_dep],
)

hikoboshi_primitive_compute_goldens = executable(
  'hikoboshi_primitive_compute_goldens',
  'tests/cpp/primitive_compute_goldens.cpp',
  dependencies: [hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_fast_exp_neg_parity_test = executable(
  'hikoboshi_fast_exp_neg_parity_test',
  'tests/parity/fast_exp_neg_parity_test.cpp',
  include_directories: hikoboshi_public_inc,
)
hikoboshi_primitive_linalg_goldens = executable(
  'hikoboshi_primitive_linalg_goldens',
  'tests/cpp/primitive_linalg_goldens.cpp',
  dependencies: [hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_primitive_alignment_goldens = executable(
  'hikoboshi_primitive_alignment_goldens',
  'tests/cpp/primitive_alignment_goldens.cpp',
  dependencies: [hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_soft_smith_waterman_scalar_tests = executable(
  'hikoboshi_soft_smith_waterman_scalar_tests',
  'tests/cpp/soft_smith_waterman_scalar_tests.cpp',
  dependencies: [hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_softmax_parity_tests = executable(
  'hikoboshi_softmax_parity_tests',
  'tests/cpp/softmax_parity_tests.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_log_softmax_parity_test = executable(
  'hikoboshi_log_softmax_parity_test',
  'tests/unit/log_softmax_parity_test.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_cpu_features_smoke = executable(
  'hikoboshi_cpu_features_smoke',
  'tests/cpp/cpu_features_smoke.cpp',
  dependencies: [hikoboshi_dispatch_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_dispatch_table_smoke = executable(
  'hikoboshi_dispatch_table_smoke',
  'tests/cpp/dispatch_table_smoke.cpp',
  dependencies: [hikoboshi_dispatch_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_dispatch_registry_smoke = executable(
  'hikoboshi_dispatch_registry_smoke',
  'tests/cpp/dispatch_registry_smoke.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_universal_dep],
)
hikoboshi_primitive_op_registry_tests = executable(
  'hikoboshi_primitive_op_registry_tests',
  'tests/cpp/primitive_op_registry_tests.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_universal_dep],
)
hikoboshi_module_op_registry_tests = executable(
  'hikoboshi_module_op_registry_tests',
  'tests/cpp/module_op_registry_tests.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_universal_dep],
  link_whole: [hikoboshi_modules_lib],
)
hikoboshi_thread_pool_smoke = executable(
  'hikoboshi_thread_pool_smoke',
  'tests/cpp/thread_pool_smoke.cpp',
  dependencies: [hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_thread_pool_no_alloc = executable(
  'hikoboshi_thread_pool_no_alloc',
  'tests/cpp/thread_pool_no_alloc.cpp',
  dependencies: [hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)

hikoboshi_structure_input_tests = executable(
  'hikoboshi_structure_input_tests',
  'tests/cpp/structure_input_tests.cpp',
  dependencies: [hikoboshi_io_dep, hikoboshi_universal_dep],
)
hikoboshi_residue_template_frozen_tests = executable(
  'hikoboshi_residue_template_frozen_tests',
  'tests/cpp/residue_template_frozen_tests.cpp',
  dependencies: [hikoboshi_io_dep, hikoboshi_universal_dep],
)
hikoboshi_errors_formatter_tests = executable(
  'hikoboshi_errors_formatter_tests',
  'tests/cpp/errors_formatter_tests.cpp',
  dependencies: [hikoboshi_errors_dep, hikoboshi_universal_dep],
)
hikoboshi_mpnn_module_goldens = executable(
  'hikoboshi_mpnn_module_goldens',
  'tests/cpp/mpnn_module_goldens.cpp',
  dependencies: [hikoboshi_modules_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_gemm_parity_modes = executable(
  'hikoboshi_gemm_parity_modes',
  'tests/cpp/gemm_parity_modes.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_mpnn_input_edge_prep_tests = executable(
  'hikoboshi_mpnn_input_edge_prep_tests',
  'tests/cpp/mpnn_input_edge_prep_tests.cpp',
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_mpnn_message_passing_tests = executable(
  'hikoboshi_mpnn_message_passing_tests',
  'tests/cpp/mpnn_message_passing_tests.cpp',
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_proteinmpnn_encoder_layer_parity_test = executable(
  'hikoboshi_proteinmpnn_encoder_layer_parity_test',
  'tests/unit/proteinmpnn_encoder_layer_parity_test.cpp',
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_proteinmpnn_recovery_parity_driver = executable(
  'hikoboshi_proteinmpnn_recovery_parity_driver',
  'tests/parity/proteinmpnn_v48_020/recovery_parity_driver.cpp',
  dependencies: [
    hikoboshi_io_dep,
    hikoboshi_modules_dep,
    hikoboshi_weights_dep,
    hikoboshi_universal_dep,
  ],
)
hikoboshi_similarity_module_goldens = executable(
  'hikoboshi_similarity_module_goldens',
  'tests/cpp/similarity_module_goldens.cpp',
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_soft_smith_waterman_module_goldens = executable(
  'hikoboshi_soft_smith_waterman_module_goldens',
  'tests/cpp/soft_smith_waterman_module_goldens.cpp',
  dependencies: [hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_attention_parity_tests = executable(
  'hikoboshi_attention_parity_tests',
  'tests/cpp/attention_parity_tests.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_esm2_forward_parity_tests = executable(
  'hikoboshi_esm2_forward_parity_tests',
  'tests/cpp/esm2_forward_parity_tests.cpp',
  dependencies: [hikoboshi_dispatch_dep, hikoboshi_modules_dep, hikoboshi_universal_dep],
)
hikoboshi_sequence_route_smoke_tests = executable(
  'hikoboshi_sequence_route_smoke_tests',
  'tests/cpp/sequence_route_smoke_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_sequence_route_parity_tests = executable(
  'hikoboshi_sequence_route_parity_tests',
  'tests/cpp/sequence_route_parity_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_encode_direct_route_parity_tests = executable(
  'hikoboshi_encode_direct_route_parity_tests',
  'tests/cpp/encode_direct_route_parity_tests.cpp',
  dependencies: [
    hikoboshi_api_dep,
    hikoboshi_modules_dep,
    hikoboshi_weights_dep,
    hikoboshi_universal_dep,
  ],
  link_args: hikoboshi_tsan_executable_link_args + hikoboshi_mpnn_forward_wrap_link_args,
  build_by_default: hikoboshi_gnu_ld_wrap_supported,
)
hikoboshi_esm2_pytorch_real_parity = executable(
  'hikoboshi_esm2_pytorch_real_parity',
  'tests/cpp/esm2_pytorch_real_parity.cpp',
  dependencies: [
    hikoboshi_algorithms_dep,
    hikoboshi_modules_dep,
    hikoboshi_weights_dep,
    hikoboshi_universal_dep,
  ],
)
hikoboshi_softsw_posterior_pytorch_parity = executable(
  'hikoboshi_softsw_posterior_pytorch_parity',
  'tests/cpp/softsw_posterior_pytorch_parity.cpp',
  dependencies: [hikoboshi_primitives_dep, hikoboshi_universal_dep],
)
hikoboshi_similarity_scalar_pytorch_parity = executable(
  'hikoboshi_similarity_scalar_pytorch_parity',
  'tests/cpp/similarity_scalar_pytorch_parity.cpp',
  dependencies: [
    hikoboshi_algorithms_dep,
    hikoboshi_modules_dep,
    hikoboshi_weights_dep,
    hikoboshi_universal_dep,
  ],
)
hikoboshi_esm2_softsw_pytorch_parity = executable(
  'esm2_softsw_pytorch_parity',
  'tests/cpp/esm2_softsw_pytorch_parity.cpp',
  dependencies: [
    hikoboshi_algorithms_dep,
    hikoboshi_modules_dep,
    hikoboshi_primitives_dep,
    hikoboshi_weights_dep,
    hikoboshi_universal_dep,
  ],
)
hikoboshi_esm2_8m_pairwise_smoke = executable(
  'hikoboshi_esm2_8m_pairwise_smoke',
  'tests/integration/esm2_8m_pairwise_smoke.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_esm2_8m_all_vs_all_smoke = executable(
  'hikoboshi_esm2_8m_all_vs_all_smoke',
  'tests/integration/esm2_8m_all_vs_all_smoke.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_pair_list_smoke = executable(
  'hikoboshi_pair_list_smoke',
  'tests/integration/pair_list_smoke.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_pair_list_vs_pairwise_parity = executable(
  'hikoboshi_pair_list_vs_pairwise_parity',
  'tests/integration/pair_list_vs_pairwise_parity.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_pair_list_cache_hit = executable(
  'hikoboshi_pair_list_cache_hit',
  'tests/integration/pair_list_cache_hit.cpp',
  dependencies: [
    hikoboshi_api_dep,
    hikoboshi_algorithms_dep,
    hikoboshi_weights_dep,
    hikoboshi_universal_dep,
  ],
)
hikoboshi_pairwise_workspace_tests = executable(
  'hikoboshi_pairwise_workspace_tests',
  'tests/cpp/pairwise_workspace_tests.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
)
hikoboshi_resolved_alignment_problem_tests = executable(
  'hikoboshi_resolved_alignment_problem_tests',
  'tests/cpp/resolved_alignment_problem_tests.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
)
hikoboshi_metrics_tests = executable(
  'hikoboshi_metrics_tests',
  'tests/cpp/metrics_tests.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
)
hikoboshi_api_engine_smoke = executable(
  'hikoboshi_api_engine_smoke',
  'tests/cpp/api_engine_smoke.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_api_pairwise_soft_smoke = executable(
  'hikoboshi_api_pairwise_soft_smoke',
  'tests/cpp/api_pairwise_soft_smoke.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_universal_dep],
)
hikoboshi_api_backend_capabilities_tests = executable(
  'hikoboshi_api_backend_capabilities_tests',
  'tests/cpp/api_backend_capabilities_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_universal_dep],
)
hikoboshi_pairwise_pipeline_tests = executable(
  'hikoboshi_pairwise_pipeline_tests',
  'tests/cpp/pairwise_pipeline_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_score_alignment_tests = executable(
  'hikoboshi_score_alignment_tests',
  'tests/cpp/api/score_alignment_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
)
hikoboshi_pairwise_pipeline_soft_tests = executable(
  'hikoboshi_pairwise_pipeline_soft_tests',
  'tests/cpp/pairwise_pipeline_soft_tests.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
)
hikoboshi_all_vs_all_pair_order_tests = executable(
  'hikoboshi_all_vs_all_pair_order_tests',
  'tests/cpp/all_vs_all_pair_order_tests.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args + hikoboshi_mpnn_forward_wrap_link_args,
  build_by_default: hikoboshi_gnu_ld_wrap_supported,
)
hikoboshi_all_vs_all_soft_smoke = executable(
  'hikoboshi_all_vs_all_soft_smoke',
  'tests/cpp/all_vs_all_soft_smoke.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_universal_dep],
)
hikoboshi_all_vs_all_query_tiled_scheduler_tests = executable(
  'hikoboshi_all_vs_all_query_tiled_scheduler_tests',
  'tests/cpp/all_vs_all_query_tiled_scheduler_tests.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_all_vs_all_streaming_sink_tests = executable(
  'hikoboshi_all_vs_all_streaming_sink_tests',
  'tests/cpp/all_vs_all_streaming_sink_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_all_vs_all_streaming_sink_large_scale_tests = executable(
  'hikoboshi_all_vs_all_streaming_sink_large_scale_tests',
  'tests/cpp/all_vs_all_streaming_sink_large_scale_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_all_vs_all_streaming_sink_byte_identity_test = executable(
  'hikoboshi_all_vs_all_streaming_sink_byte_identity_test',
  'tests/cpp/all_vs_all_streaming_sink_byte_identity_test.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_all_vs_all_streaming_sink_stall_recovery_test = executable(
  'hikoboshi_all_vs_all_streaming_sink_stall_recovery_test',
  'tests/cpp/all_vs_all_streaming_sink_stall_recovery_test.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_all_vs_all_streaming_sink_variance_stress_test = executable(
  'hikoboshi_all_vs_all_streaming_sink_variance_stress_test',
  'tests/cpp/all_vs_all_streaming_sink_variance_stress_test.cpp',
  dependencies: [hikoboshi_algorithms_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_cli_all_vs_all_streaming_summary_tests = executable(
  'hikoboshi_cli_all_vs_all_streaming_summary_tests',
  'tests/cpp/cli_all_vs_all_streaming_summary_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_universal_dep],
)
hikoboshi_all_vs_all_parallel_correctness_tests = executable(
  'hikoboshi_all_vs_all_parallel_correctness_tests',
  'tests/cpp/all_vs_all_parallel_correctness_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_all_vs_all_parallel_encoding_tests = executable(
  'hikoboshi_all_vs_all_parallel_encoding_tests',
  'tests/cpp/all_vs_all_parallel_encoding_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args + hikoboshi_mpnn_forward_wrap_link_args,
  build_by_default: hikoboshi_gnu_ld_wrap_supported,
)
hikoboshi_pairwise_parallel_encode_tests = executable(
  'hikoboshi_pairwise_parallel_encode_tests',
  'tests/cpp/pairwise_parallel_encode_tests.cpp',
  dependencies: [hikoboshi_api_dep, hikoboshi_algorithms_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
  link_args: hikoboshi_tsan_executable_link_args,
)
hikoboshi_fasta_writer_tests = executable(
  'hikoboshi_fasta_writer_tests',
  'tests/cpp/fasta_writer_tests.cpp',
  dependencies: [hikoboshi_io_dep, hikoboshi_api_dep, hikoboshi_universal_dep],
)
hikoboshi_pdb_writer_tests = executable(
  'hikoboshi_pdb_writer_tests',
  'tests/cpp/pdb_writer_tests.cpp',
  dependencies: [hikoboshi_io_dep, hikoboshi_api_dep, hikoboshi_universal_dep],
)
hikoboshi_all_vs_all_artifact_layout_tests = executable(
  'hikoboshi_all_vs_all_artifact_layout_tests',
  'tests/cpp/all_vs_all_artifact_layout_tests.cpp',
  dependencies: [hikoboshi_io_dep, hikoboshi_universal_dep],
)
hikoboshi_performance_smoke = executable(
  'hikoboshi_performance_smoke',
  'tests/cpp/performance_smoke.cpp',
  dependencies: [
    hikoboshi_api_dep,
    hikoboshi_algorithms_dep,
    hikoboshi_modules_dep,
    hikoboshi_weights_dep,
    hikoboshi_universal_dep,
  ],
)
if hikoboshi_python_api
  hikoboshi_bindings_helpers_smoke = executable(
    'hikoboshi_bindings_helpers_smoke',
    'tests/cpp/bindings_helpers_smoke.cpp',
    dependencies: [python_embed_dep, hikoboshi_universal_dep],
    include_directories: hikoboshi_public_inc,
  )
  hikoboshi_bindings_casters_tests = executable(
    'hikoboshi_bindings_casters_tests',
    'tests/cpp/bindings_casters_tests.cpp',
    dependencies: [python_embed_dep],
    include_directories: hikoboshi_public_inc,
  )
  hikoboshi_bindings_args_tests = executable(
    'hikoboshi_bindings_args_tests',
    'tests/cpp/bindings_args_tests.cpp',
    dependencies: [python_embed_dep],
    include_directories: hikoboshi_public_inc,
  )
  hikoboshi_dlpack_compat_tests = executable(
    'hikoboshi_dlpack_compat_tests',
    'tests/cpp/dlpack_compat_tests.cpp',
    dependencies: [python_embed_dep],
    include_directories: hikoboshi_public_inc,
  )
endif
if hikoboshi_bench_mpnn_dump
  hikoboshi_mpnn64_real_parity = executable(
    'hikoboshi_mpnn64_real_parity',
    'tests/cpp/mpnn64_real_parity.cpp',
    dependencies: [hikoboshi_modules_dep, hikoboshi_weights_dep, hikoboshi_universal_dep],
  )
endif

hikoboshi_build_root_env = ['HIKOBOSHI_BUILD_ROOT=' + meson.current_build_dir()]
hikoboshi_source_root_env = ['HIKOBOSHI_SOURCE_ROOT=' + meson.project_source_root()]

test('hikoboshi-cli-placeholder', hikoboshi_cli, args: ['--version'])
if hikoboshi_python_api
  test(
    'hikoboshi-cli-install-name-placeholder',
    python,
    args: [files('tests/python/run_build_cli_alias.py')],
    depends: [hikoboshi_cli_install_name],
    env: hikoboshi_build_root_env,
  )
endif

hikoboshi_plain_tests = [
  ['hikoboshi-target-graph-smoke', hikoboshi_target_graph_smoke],
  ['hikoboshi-alignment-path-smoke', hikoboshi_alignment_path_smoke],
  ['hikoboshi-universal-public-headers', hikoboshi_universal_public_headers],
  ['hikoboshi-reserved-workspace-types-smoke', hikoboshi_reserved_workspace_types_smoke],
  ['hikoboshi-api-public-headers', hikoboshi_api_public_headers],
  ['hikoboshi-weights-public-headers', hikoboshi_weights_public_headers],
]

foreach case : hikoboshi_plain_tests
  test(case[0], case[1])
endforeach

hikoboshi_suite_tests = [
  ['hikoboshi-proteinmpnn-v48-020-contract-test', hikoboshi_proteinmpnn_v48_020_contract_test, ['universal', 'contract']],
  ['hikoboshi-proteinmpnn-decoder-layer-parity-test', hikoboshi_proteinmpnn_decoder_layer_parity_test, ['modules', 'parity']],
  ['hikoboshi-primitive-compute-goldens', hikoboshi_primitive_compute_goldens, 'primitives'],
  ['hikoboshi-fast-exp-neg-parity-test', hikoboshi_fast_exp_neg_parity_test, ['primitives', 'parity']],
  ['hikoboshi-primitive-linalg-goldens', hikoboshi_primitive_linalg_goldens, 'primitives'],
  ['hikoboshi-primitive-alignment-goldens', hikoboshi_primitive_alignment_goldens, 'primitives'],
  ['hikoboshi-gemm-parity-modes', hikoboshi_gemm_parity_modes, ['dispatch', 'parity']],
  ['hikoboshi-soft-smith-waterman-scalar-tests', hikoboshi_soft_smith_waterman_scalar_tests, ['primitives', 'parity']],
  ['hikoboshi-softmax-parity-tests', hikoboshi_softmax_parity_tests, ['primitives', 'parity']],
  ['hikoboshi-log-softmax-parity-test', hikoboshi_log_softmax_parity_test, ['primitives', 'parity']],
  ['hikoboshi-cpu-features-smoke', hikoboshi_cpu_features_smoke, 'dispatch'],
  ['hikoboshi-dispatch-table-smoke', hikoboshi_dispatch_table_smoke, 'dispatch'],
  ['hikoboshi-dispatch-registry-smoke', hikoboshi_dispatch_registry_smoke, 'dispatch'],
  ['hikoboshi-primitive-op-registry-tests', hikoboshi_primitive_op_registry_tests, 'dispatch'],
  ['hikoboshi-module-op-registry-tests', hikoboshi_module_op_registry_tests, 'dispatch'],
  ['hikoboshi-thread-pool-smoke', hikoboshi_thread_pool_smoke, 'dispatch'],
  ['hikoboshi-thread-pool-no-alloc', hikoboshi_thread_pool_no_alloc, 'dispatch'],
  ['hikoboshi-structure-input-tests', hikoboshi_structure_input_tests, 'io'],
  ['hikoboshi-residue-template-frozen-tests', hikoboshi_residue_template_frozen_tests, 'io'],
  ['hikoboshi-errors-formatter-tests', hikoboshi_errors_formatter_tests, 'errors'],
  ['hikoboshi-mpnn-module-goldens', hikoboshi_mpnn_module_goldens, 'modules'],
  ['hikoboshi-mpnn-input-edge-prep-tests', hikoboshi_mpnn_input_edge_prep_tests, 'modules'],
  ['hikoboshi-mpnn-message-passing-tests', hikoboshi_mpnn_message_passing_tests, 'modules'],
  ['hikoboshi-proteinmpnn-encoder-layer-parity-test', hikoboshi_proteinmpnn_encoder_layer_parity_test, ['modules', 'parity']],
  ['hikoboshi-similarity-module-goldens', hikoboshi_similarity_module_goldens, 'modules'],
  ['hikoboshi-soft-smith-waterman-module-goldens', hikoboshi_soft_smith_waterman_module_goldens, ['modules', 'parity']],
  ['hikoboshi-attention-parity-tests', hikoboshi_attention_parity_tests, ['modules', 'parity']],
  ['hikoboshi-esm2-forward-parity-tests', hikoboshi_esm2_forward_parity_tests, ['modules', 'parity']],
  ['hikoboshi-sequence-route-smoke-tests', hikoboshi_sequence_route_smoke_tests, ['api', 'sequence-route']],
  ['hikoboshi-sequence-route-parity-tests', hikoboshi_sequence_route_parity_tests, ['api', 'sequence-route', 'parity']],
  ['hikoboshi-esm2-8m-pairwise-smoke', hikoboshi_esm2_8m_pairwise_smoke, ['integration', 'esm2-8m']],
  ['hikoboshi-esm2-8m-all-vs-all-smoke', hikoboshi_esm2_8m_all_vs_all_smoke, ['integration', 'esm2-8m']],
  ['hikoboshi-pair-list-smoke', hikoboshi_pair_list_smoke, ['integration']],
  ['hikoboshi-pair-list-vs-pairwise-parity', hikoboshi_pair_list_vs_pairwise_parity, ['integration']],
  ['hikoboshi-pair-list-cache-hit', hikoboshi_pair_list_cache_hit, ['integration']],
  ['hikoboshi-pairwise-workspace-tests', hikoboshi_pairwise_workspace_tests, 'algorithms'],
  ['hikoboshi-resolved-alignment-problem-tests', hikoboshi_resolved_alignment_problem_tests, 'algorithms'],
  ['hikoboshi-metrics-tests', hikoboshi_metrics_tests, 'algorithms'],
  ['hikoboshi-api-engine-smoke', hikoboshi_api_engine_smoke, 'api'],
  ['hikoboshi-api-pairwise-soft-smoke', hikoboshi_api_pairwise_soft_smoke, 'api'],
  ['hikoboshi-api-backend-capabilities-tests', hikoboshi_api_backend_capabilities_tests, 'api'],
  ['hikoboshi-pairwise-pipeline-tests', hikoboshi_pairwise_pipeline_tests, 'api'],
  ['hikoboshi-score-alignment-tests', hikoboshi_score_alignment_tests, 'api'],
  ['hikoboshi-pairwise-pipeline-soft-tests', hikoboshi_pairwise_pipeline_soft_tests, ['algorithms', 'parity']],
  ['hikoboshi-all-vs-all-soft-smoke', hikoboshi_all_vs_all_soft_smoke, ['api', 'all-vs-all']],
  ['hikoboshi-all-vs-all-query-tiled-scheduler-tests', hikoboshi_all_vs_all_query_tiled_scheduler_tests, ['algorithms', 'all-vs-all']],
  ['hikoboshi-all-vs-all-streaming-sink-tests', hikoboshi_all_vs_all_streaming_sink_tests, ['algorithms', 'all-vs-all']],
  ['hikoboshi-all-vs-all-streaming-sink-large-scale-tests', hikoboshi_all_vs_all_streaming_sink_large_scale_tests, ['algorithms', 'all-vs-all']],
  ['hikoboshi-all-vs-all-streaming-sink-byte-identity-test', hikoboshi_all_vs_all_streaming_sink_byte_identity_test, ['algorithms', 'all-vs-all']],
  ['hikoboshi-all-vs-all-streaming-sink-stall-recovery-test', hikoboshi_all_vs_all_streaming_sink_stall_recovery_test, ['algorithms', 'all-vs-all']],
  ['hikoboshi-all-vs-all-streaming-sink-variance-stress-test', hikoboshi_all_vs_all_streaming_sink_variance_stress_test, ['algorithms', 'all-vs-all', 'performance', 'perf']],
  ['hikoboshi-cli-all-vs-all-streaming-summary-tests', hikoboshi_cli_all_vs_all_streaming_summary_tests, ['parity', 'all-vs-all']],
  ['hikoboshi-all-vs-all-parallel-correctness-tests', hikoboshi_all_vs_all_parallel_correctness_tests, ['algorithms', 'all-vs-all']],
  ['hikoboshi-pairwise-parallel-encode-tests', hikoboshi_pairwise_parallel_encode_tests, ['algorithms', 'api']],
  ['hikoboshi-fasta-writer-tests', hikoboshi_fasta_writer_tests, 'io-artifacts'],
  ['hikoboshi-pdb-writer-tests', hikoboshi_pdb_writer_tests, 'io-artifacts'],
  ['hikoboshi-all-vs-all-artifact-layout-tests', hikoboshi_all_vs_all_artifact_layout_tests, 'io-artifacts'],
  ['hikoboshi-performance-smoke', hikoboshi_performance_smoke, ['performance', 'perf']],
  ['hikoboshi-weights-package-provider-tests', hikoboshi_weights_package_provider_tests, 'weights'],
  ['hikoboshi-tensor-slot-smoke', hikoboshi_tensor_slot_smoke, 'weights'],
  ['hikoboshi-package-registry-tests', hikoboshi_package_registry_tests, 'weights'],
  ['hikoboshi-package-validation-pipeline-tests', hikoboshi_package_validation_pipeline_tests, ['weights', 'dispatch']],
  ['hikoboshi-package-validation-noalloc-test', hikoboshi_package_validation_noalloc_test, ['weights', 'dispatch']],
  ['hikoboshi-api-default-package-provider-regression', hikoboshi_weights_package_provider_tests, 'api'],
  ['test_dispatch_seam', hikoboshi_core_dispatch_seam_test, 'core'],
]

if hikoboshi_python_api
  hikoboshi_suite_tests += [
    ['hikoboshi-bindings-helpers-smoke', hikoboshi_bindings_helpers_smoke, 'bindings'],
    ['hikoboshi-bindings-casters-tests', hikoboshi_bindings_casters_tests, 'bindings'],
    ['hikoboshi-bindings-args-tests', hikoboshi_bindings_args_tests, 'bindings'],
    ['hikoboshi-dlpack-compat-tests', hikoboshi_dlpack_compat_tests, 'bindings'],
  ]
endif

if hikoboshi_gnu_ld_wrap_supported
  hikoboshi_suite_tests += [
    ['hikoboshi-encode-direct-route-parity-tests', hikoboshi_encode_direct_route_parity_tests, ['cpp', 'api', 'parity']],
    ['hikoboshi-all-vs-all-pair-order-tests', hikoboshi_all_vs_all_pair_order_tests, ['algorithms', 'all-vs-all']],
    ['hikoboshi-all-vs-all-parallel-encoding-tests', hikoboshi_all_vs_all_parallel_encoding_tests, ['algorithms', 'all-vs-all']],
  ]
endif

hikoboshi_tsan_dispatch_test_names = [
  'hikoboshi-cpu-features-smoke',
  'hikoboshi-dispatch-table-smoke',
  'hikoboshi-thread-pool-smoke',
  'hikoboshi-thread-pool-no-alloc',
  'hikoboshi-all-vs-all-pair-order-tests',
  'hikoboshi-all-vs-all-query-tiled-scheduler-tests',
  'hikoboshi-all-vs-all-streaming-sink-tests',
  'hikoboshi-all-vs-all-streaming-sink-large-scale-tests',
  'hikoboshi-all-vs-all-streaming-sink-byte-identity-test',
  'hikoboshi-all-vs-all-streaming-sink-stall-recovery-test',
  'hikoboshi-all-vs-all-streaming-sink-variance-stress-test',
  'hikoboshi-all-vs-all-parallel-correctness-tests',
  'hikoboshi-all-vs-all-parallel-encoding-tests',
  'hikoboshi-pairwise-parallel-encode-tests',
]

# Tests that measure real wall-clock latency and would fail under
# unit-test-runner parallelism (other tests compete for CPU and disturb
# the per-pair busy-wait granularity). Run these without parallelism so
# the asserted timing bounds reflect the streaming-sink overhead, not
# CPU contention with sibling tests.
hikoboshi_serial_only_test_names = [
  'hikoboshi-all-vs-all-streaming-sink-variance-stress-test',
]

foreach case : hikoboshi_suite_tests
  hikoboshi_tsan_wrap_case = hikoboshi_build_sanitize == 'thread'
  hikoboshi_tsan_wrap_case = hikoboshi_tsan_wrap_case and (
    case[0] in hikoboshi_tsan_dispatch_test_names
  )
  hikoboshi_tsan_wrap_case = hikoboshi_tsan_wrap_case and hikoboshi_tsan_setarch_found
  hikoboshi_serial_only_case = case[0] in hikoboshi_serial_only_test_names
  if hikoboshi_tsan_wrap_case
    test(
      case[0],
      hikoboshi_tsan_setarch,
      args: [hikoboshi_tsan_setarch_arch, '-R', case[1]],
      depends: [case[1]],
      suite: case[2],
      is_parallel: not hikoboshi_serial_only_case,
    )
  else
    test(
      case[0],
      case[1],
      suite: case[2],
      is_parallel: not hikoboshi_serial_only_case,
    )
  endif
endforeach

if hikoboshi_python_api
test(
  'hikoboshi-no-dev-paths-in-binaries',
  python,
  args: [files('tests/python/test_no_dev_paths_in_binary.py')],
  depends: [hikoboshi_pyext_module, hikoboshi_weights_lib],
  env: hikoboshi_build_root_env,
  suite: 'weights',
)
test(
  'hikoboshi-python-extension-placeholder',
  python,
  args: [files('tests/python/import_extension_placeholder.py')],
  depends: [hikoboshi_pyext_module],
  env: hikoboshi_build_root_env,
)
test(
  'hikoboshi-python-package-placeholder',
  python,
  args: [files('tests/python/import_package_placeholder.py')],
  env: ['PYTHONPATH=' + meson.project_source_root() / 'python'],
)
hikoboshi_python_test_env = hikoboshi_build_root_env + [
  'PYTHONPATH=' + meson.project_source_root() / 'python',
]

hikoboshi_python_tests = [
  ['hikoboshi-python-imports', 'tests/python/test_imports.py'],
  ['hikoboshi-python-info', 'tests/python/test_info.py'],
  ['hikoboshi-python-info-backends', 'tests/python/test_info_backends.py'],
  ['hikoboshi-python-info-models', 'tests/python/test_info_models.py'],
  ['hikoboshi-python-capability-introspection', 'tests/python/test_capability_introspection.py'],
  ['hikoboshi-python-callable-namespaces', 'tests/python/test_callable_namespaces.py'],
  ['hikoboshi-python-package-kwarg', 'tests/python/test_package_kwarg.py'],
  ['hikoboshi-python-encode-smoke', 'tests/python/test_encode_smoke.py'],
  ['hikoboshi-python-structure-input-smoke', 'tests/python/test_structure_input_smoke.py'],
  ['hikoboshi-python-numpy-dtype', 'tests/python/test_numpy_dtype.py'],
  ['hikoboshi-python-tensor-buffer-protocol', 'tests/python/test_tensor_buffer_protocol.py'],
  ['hikoboshi-python-tensor-dlpack', 'tests/python/test_tensor_dlpack.py'],
  ['hikoboshi-python-tensor-array-protocol', 'tests/python/test_tensor_array_protocol.py'],
  ['hikoboshi-python-embedding-only-metadata', 'tests/python/test_embedding_only_metadata.py'],
  ['hikoboshi-python-exceptions', 'tests/python/test_exceptions.py'],
  ['hikoboshi-python-no-legacy-symbols', 'tests/python/test_no_legacy_symbols.py'],
  ['hikoboshi-python-all-vs-all-pair-order', 'tests/python/test_all_vs_all_pair_order.py'],
  ['hikoboshi-python-all-vs-all-thread-count', 'tests/python/test_all_vs_all_thread_count.py'],
  ['hikoboshi-python-all-vs-all-streaming-to-tsv', 'tests/python/test_all_vs_all_streaming_to_tsv.py'],
  ['hikoboshi-python-pair-list', 'tests/python/test_pair_list.py'],
  ['hikoboshi-python-pybind-extension-name', 'tests/python/test_pybind_extension_name.py'],
  ['hikoboshi-python-pairwise-soft', 'tests/python/test_pairwise_soft.py'],
  ['hikoboshi-python-all-vs-all-soft', 'tests/python/test_all_vs_all_soft.py'],
  ['hikoboshi-python-score-alignment', 'tests/python/test_score_alignment.py'],
  ['hikoboshi-python-sequence-route-smoke', 'tests/python/test_sequence_route_smoke.py'],
  ['hikoboshi-python-inverse-fold-binding', 'tests/python/inverse_fold_binding_test.py'],
]

foreach case : hikoboshi_python_tests
  test(
    case[0],
    python,
    args: [files(case[1])],
    depends: [hikoboshi_pyext_module],
    env: hikoboshi_python_test_env,
    suite: 'python',
  )
endforeach

test(
  'hikoboshi-pair-list-perf-smoke',
  python,
  args: [files('tests/perf/run_pair_list_perf_smoke.py')],
  depends: [hikoboshi_pyext_module],
  env: hikoboshi_python_test_env + hikoboshi_source_root_env + [
    'HIKOBOSHI_PAIR_LIST_PERF_JSON=@0@'.format(
      meson.current_build_dir() / 'pair_list_perf_smoke.json',
    ),
  ],
  suite: ['performance', 'perf'],
  timeout: 180,
)

test(
  'hikoboshi-python-no-numpy',
  python,
  args: [files('tests/python/test_no_numpy.py')],
  depends: [hikoboshi_pyext_module],
  env: hikoboshi_python_test_env,
  suite: 'python-no-numpy',
)

hikoboshi_pytorch_parity_env = hikoboshi_python_test_env + hikoboshi_source_root_env
test(
  'hikoboshi-proteinmpnn-v48-020-recovery-parity',
  python,
  args: [
    files('tests/parity/proteinmpnn_v48_020/recovery_parity.py'),
    meson.current_build_dir() / 'hikoboshi_proteinmpnn_recovery_parity_driver',
  ],
  depends: [hikoboshi_proteinmpnn_recovery_parity_driver],
  env: hikoboshi_source_root_env,
  suite: 'parity',
  timeout: 120,
)
test(
  'hikoboshi-proteinmpnn-v48-020-encoder-layer-pytorch-parity',
  python,
  args: [files('tests/parity/proteinmpnn_v48_020/encoder_layer_parity.py')],
  depends: [hikoboshi_proteinmpnn_encoder_layer_parity_test],
  env: hikoboshi_pytorch_parity_env,
  suite: ['modules', 'parity-pytorch'],
  timeout: 120,
)
test(
  'hikoboshi-esm2-pytorch-real-parity',
  hikoboshi_esm2_pytorch_real_parity,
  env: hikoboshi_source_root_env,
  suite: 'parity-pytorch',
  timeout: 60,
)
test(
  'hikoboshi-proteinmpnn-decoder-layer-head-parity',
  python,
  args: [files('tests/parity/proteinmpnn_v48_020/decoder_layer_head_parity.py')],
  env: hikoboshi_source_root_env,
  suite: ['modules', 'parity', 'parity-pytorch'],
  timeout: 60,
)
# fe1 diagnostic test: feeds the PyTorch-encoder sim_matrix straight into
# Hikoboshi's soft_smith_waterman_scalar kernel and compares posterior +
# log_partition element-by-element against the orihime.soft_sw_affine
# golden. The packet hypothesis was that the kernel itself diverges from
# the orihime reference on real ESM2-derived score matrices; the test
# disproves that hypothesis -- the kernel matches PyTorch to ~3e-6
# (well under the 1e-4 fast-mode contract) on all three lead-diagnostic
# pairs. The test is left as expected-pass so it locks the kernel-side
# parity contract; fe2 must investigate the encoder + similarity layers
# upstream of the kernel rather than the kernel itself.
test(
  'hikoboshi-softsw-posterior-pytorch-parity',
  hikoboshi_softsw_posterior_pytorch_parity,
  env: hikoboshi_source_root_env,
  suite: 'parity-pytorch',
  timeout: 60,
)
# fe2 diagnostic test: runs the full Hikoboshi pairwise sequence-input
# pipeline on the three fe1 lead pairs (with CLS+EOS-wrapped tokens to
# match the PyTorch reference) and compares the `workspace.similarity_data()`
# buffer to fe1's PyTorch `sim_matrix` golden element-by-element. The
# tolerance (`2e-3`) is just above the conservative encoder-drift bound
# `sqrt(320) * 1e-4 = 1.79e-3`, so the test admits encoder drift but
# rejects any structural similarity_scalar bug. The fe2 outcome (PASS /
# FAIL per pair) is the input fe3 needs to decide between
# similarity_scalar fix (case A) and encoder/plumbing fix (case B/C);
# The tolerance admits bounded encoder drift but rejects a structural
# similarity-kernel regression.
test(
  'hikoboshi-similarity-scalar-pytorch-parity',
  hikoboshi_similarity_scalar_pytorch_parity,
  env: hikoboshi_source_root_env,
  suite: 'parity-pytorch',
  timeout: 60,
)
# fe4 per-stage parity: exercises tokenization, encoder forward,
# sim_matrix, soft_sw_affine internals (log_partition, V, M, posterior),
# and the posterior > kPairwiseSoftPosteriorThreshold path-step extraction
# on the same three fe1 lead pairs end-to-end. Any failing stage points
# fe5 at exactly the divergent stage (D1.<stage> dispatch hint printed
# on failure).
test(
  'hikoboshi-esm2-softsw-pytorch-parity',
  hikoboshi_esm2_softsw_pytorch_parity,
  env: hikoboshi_source_root_env,
  suite: 'parity-pytorch',
  timeout: 90,
)
endif

hikoboshi_cli_design_smoke_test = executable(
  'hikoboshi_cli_design_smoke_test',
  'tests/cli/design_smoke_test.cpp',
)
test(
  'hikoboshi-cli-design-smoke',
  hikoboshi_cli_design_smoke_test,
  args: [meson.current_build_dir() / 'hikoboshi'],
  depends: [hikoboshi_cli_install_name],
  suite: 'cli',
)

if hikoboshi_python_api
  hikoboshi_cli_tests = [
    ['hikoboshi-cli-pairwise-smoke', 'tests/cli/run_cli_pairwise_smoke.py'],
    ['hikoboshi-cli-pairwise-sequence-smoke', 'tests/cli/run_cli_pairwise_sequence_smoke.py'],
    ['hikoboshi-cli-pair-list-smoke', 'tests/cli/run_cli_pair_list_smoke.py'],
    ['hikoboshi-cli-all-vs-all-smoke', 'tests/cli/run_cli_all_vs_all_smoke.py'],
    ['hikoboshi-cli-score-alignment-smoke', 'tests/cli/test_score_alignment.sh'],
    ['hikoboshi-cli-invalid-metric-na', 'tests/cli/run_cli_invalid_metric_na.py'],
    ['hikoboshi-cli-default-weights', 'tests/cli/run_cli_default_weights.py'],
    ['hikoboshi-cli-package-flag', 'tests/cli/run_cli_package_flag.py'],
    ['hikoboshi-cli-backend-rejection', 'tests/cli/run_cli_backend_rejection.py'],
    ['hikoboshi-cli-info-models', 'tests/cli/run_cli_info_models.py'],
    ['hikoboshi-cli-capability-introspection', 'tests/cli/run_cli_capability_introspection.py'],
  ]

  foreach case : hikoboshi_cli_tests
    test(
      case[0],
      python,
      args: [files(case[1])],
      depends: [hikoboshi_cli_install_name],
      env: hikoboshi_build_root_env,
      suite: 'cli',
    )
  endforeach
endif
