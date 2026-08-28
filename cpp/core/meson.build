hikoboshi_core_inc = include_directories('../..')

hikoboshi_core_sources = files(
  'dispatch/op_registry.cpp',
  'universal/memory/growable_arena.cpp',
)

hikoboshi_core_lib = static_library(
  'cpp_core',
  hikoboshi_core_sources,
  include_directories: hikoboshi_core_inc,
)

hikoboshi_core_dep = declare_dependency(
  include_directories: hikoboshi_core_inc,
  link_with: hikoboshi_core_lib,
)
