#ifndef HIKOBOSHI_MODULES_MPNN_HPP
#define HIKOBOSHI_MODULES_MPNN_HPP

#include <cstddef>
#include <cstdint>

#include <hikoboshi/modules/detail/mpnn_layers.hpp>
#include <hikoboshi/modules/detail/mpnn_workspace.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::modules {

struct Mpnn64Descriptor {
  std::size_t hidden_dimension;
  std::size_t neighbor_count;
  std::size_t rbf_count;
  std::size_t layer_count;
  float message_scale;
};

#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
enum class Mpnn64IntermediateDtype : std::uint8_t {
  Float32 = 0,
  Int32 = 1,
};

struct Mpnn64IntermediateTensor {
  const char* name;
  const void* data;
  Mpnn64IntermediateDtype dtype;
  std::size_t rank;
  std::size_t shape[4];
};

using Mpnn64IntermediateDumpCallback =
    void (*)(const Mpnn64IntermediateTensor& tensor, void* user_data) noexcept;

struct Mpnn64IntermediateDumper {
  Mpnn64IntermediateDumpCallback callback = nullptr;
  void* user_data = nullptr;
};
#endif

struct Mpnn64DebugCapture {
  // [L, K, hidden_dimension] W_e-projected edge embedding state.
  float* edge_embedding_output = nullptr;
  // [L, hidden_dimension] node state after each encoder layer boundary.
  float* message_layer_0_output = nullptr;
  float* message_layer_1_output = nullptr;
  float* message_layer_2_output = nullptr;
  // [L, hidden_dimension] final encoder output.
  float* final_encoder_output = nullptr;
};

struct Mpnn64ForwardRequest {
  const float* coordinates;  // row-major [L, 5, 3]
  const hikoboshi::universal::AtomSource* atom_sources;  // row-major [L, 5]
  std::size_t residue_count;
  Mpnn64Descriptor descriptor;
  const detail::Mpnn64Weights* weights;
  detail::Mpnn64Workspace* workspace;
  const std::int32_t* residue_indices = nullptr;  // optional [L], defaults to 0..L-1
  const std::int32_t* chain_labels = nullptr;     // optional [L], defaults to one chain
  Mpnn64DebugCapture* debug_capture = nullptr;
#ifdef HIKOBOSHI_BENCH_MPNN_DUMP
  Mpnn64IntermediateDumper intermediate_dumper{};
#endif
};

struct Mpnn64ForwardOutput {
  float* embeddings;  // row-major [L, 64]
  std::size_t residue_count;
  std::size_t hidden_dimension;
};

hikoboshi::universal::Status mpnn64_forward_scalar(
    const Mpnn64ForwardRequest& request,
    const Mpnn64ForwardOutput& output) noexcept;

}  // namespace hikoboshi::modules

#endif  // HIKOBOSHI_MODULES_MPNN_HPP
