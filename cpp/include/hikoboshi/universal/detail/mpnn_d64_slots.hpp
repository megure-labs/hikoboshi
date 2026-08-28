#ifndef HIKOBOSHI_UNIVERSAL_DETAIL_MPNN_D64_SLOTS_HPP
#define HIKOBOSHI_UNIVERSAL_DETAIL_MPNN_D64_SLOTS_HPP

/// @file
/// Static tensor-slot metadata for the compiled hikoboshi-mpnn-d64 package.

#include <hikoboshi/universal/detail/tensor_slot.hpp>

#include <cstddef>
#include <string_view>

namespace hikoboshi::universal::detail {

/// Metadata for one runtime tensor slot in the compiled hikoboshi-mpnn-d64 blob.
///
/// `data_offset` and `byte_length` describe the float32 payload slice. `shape`
/// points to static extent storage owned by the matching `Mpnn64Slot`.
struct Mpnn64SlotInfo {
  std::string_view name;
  std::size_t data_offset;
  const std::size_t* shape;
  std::size_t rank;
  std::size_t element_count;
  std::size_t byte_length;
};

/// Generated slot table for tensors consumed by the 0.1.0 runtime path.
///
/// Entries are kept in payload order so descriptor validation can compare
/// names, offsets, shapes, and byte ranges against the compiled manifest.
#define HIKOBOSHI_MPNN64_RUNTIME_SLOT_TABLE(HIKOBOSHI_MPNN64_SLOT) \
  HIKOBOSHI_MPNN64_SLOT(w_e_bias, "W_e.bias", 0, 64) \
  HIKOBOSHI_MPNN64_SLOT(w_e_weight, "W_e.weight", 256, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(edge_embedding_norm_bias, "edge_embedding.norm.bias", 16640, 64) \
  HIKOBOSHI_MPNN64_SLOT(edge_embedding_norm_weight, "edge_embedding.norm.weight", 16896, 64) \
  HIKOBOSHI_MPNN64_SLOT(edge_embedding_weight, "edge_embedding.weight", 17152, 64,416) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w1_bias, "layers.0.W1.bias", 123656, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w1_weight, "layers.0.W1.weight", 123912, 64,192) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w11_bias, "layers.0.W11.bias", 173064, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w11_weight, "layers.0.W11.weight", 173320, 64,192) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w12_bias, "layers.0.W12.bias", 222472, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w12_weight, "layers.0.W12.weight", 222728, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w13_bias, "layers.0.W13.bias", 239112, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w13_weight, "layers.0.W13.weight", 239368, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w2_bias, "layers.0.W2.bias", 255752, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w2_weight, "layers.0.W2.weight", 256008, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w3_bias, "layers.0.W3.bias", 272392, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_w3_weight, "layers.0.W3.weight", 272648, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_ffn_w_in_bias, "layers.0.ffn.W_in.bias", 289032, 256) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_ffn_w_in_weight, "layers.0.ffn.W_in.weight", 290056, 256,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_ffn_w_out_bias, "layers.0.ffn.W_out.bias", 355592, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_ffn_w_out_weight, "layers.0.ffn.W_out.weight", 355848, 64,256) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_norm1_bias, "layers.0.norm1.bias", 421384, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_norm1_weight, "layers.0.norm1.weight", 421640, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_norm2_bias, "layers.0.norm2.bias", 421896, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_norm2_weight, "layers.0.norm2.weight", 422152, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_norm3_bias, "layers.0.norm3.bias", 422408, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_0_norm3_weight, "layers.0.norm3.weight", 422664, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w1_bias, "layers.1.W1.bias", 422920, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w1_weight, "layers.1.W1.weight", 423176, 64,192) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w11_bias, "layers.1.W11.bias", 472328, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w11_weight, "layers.1.W11.weight", 472584, 64,192) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w12_bias, "layers.1.W12.bias", 521736, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w12_weight, "layers.1.W12.weight", 521992, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w13_bias, "layers.1.W13.bias", 538376, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w13_weight, "layers.1.W13.weight", 538632, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w2_bias, "layers.1.W2.bias", 555016, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w2_weight, "layers.1.W2.weight", 555272, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w3_bias, "layers.1.W3.bias", 571656, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_w3_weight, "layers.1.W3.weight", 571912, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_ffn_w_in_bias, "layers.1.ffn.W_in.bias", 588296, 256) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_ffn_w_in_weight, "layers.1.ffn.W_in.weight", 589320, 256,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_ffn_w_out_bias, "layers.1.ffn.W_out.bias", 654856, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_ffn_w_out_weight, "layers.1.ffn.W_out.weight", 655112, 64,256) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_norm1_bias, "layers.1.norm1.bias", 720648, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_norm1_weight, "layers.1.norm1.weight", 720904, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_norm2_bias, "layers.1.norm2.bias", 721160, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_norm2_weight, "layers.1.norm2.weight", 721416, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_norm3_bias, "layers.1.norm3.bias", 721672, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_1_norm3_weight, "layers.1.norm3.weight", 721928, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w1_bias, "layers.2.W1.bias", 722184, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w1_weight, "layers.2.W1.weight", 722440, 64,192) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w11_bias, "layers.2.W11.bias", 771592, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w11_weight, "layers.2.W11.weight", 771848, 64,192) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w12_bias, "layers.2.W12.bias", 821000, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w12_weight, "layers.2.W12.weight", 821256, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w13_bias, "layers.2.W13.bias", 837640, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w13_weight, "layers.2.W13.weight", 837896, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w2_bias, "layers.2.W2.bias", 854280, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w2_weight, "layers.2.W2.weight", 854536, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w3_bias, "layers.2.W3.bias", 870920, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_w3_weight, "layers.2.W3.weight", 871176, 64,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_ffn_w_in_bias, "layers.2.ffn.W_in.bias", 887560, 256) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_ffn_w_in_weight, "layers.2.ffn.W_in.weight", 888584, 256,64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_ffn_w_out_bias, "layers.2.ffn.W_out.bias", 954120, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_ffn_w_out_weight, "layers.2.ffn.W_out.weight", 954376, 64,256) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_norm1_bias, "layers.2.norm1.bias", 1019912, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_norm1_weight, "layers.2.norm1.weight", 1020168, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_norm2_bias, "layers.2.norm2.bias", 1020424, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_norm2_weight, "layers.2.norm2.weight", 1020680, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_norm3_bias, "layers.2.norm3.bias", 1020936, 64) \
  HIKOBOSHI_MPNN64_SLOT(layers_2_norm3_weight, "layers.2.norm3.weight", 1021192, 64) \
  HIKOBOSHI_MPNN64_SLOT(positional_encoding_bias, "positional_encoding.bias", 1021448, 16) \
  HIKOBOSHI_MPNN64_SLOT(positional_encoding_weight, "positional_encoding.weight", 1021512, 16,66)

enum class Mpnn64SlotId : std::size_t {
#define HIKOBOSHI_MPNN64_ENUM(id, name, offset, ...) id,
  HIKOBOSHI_MPNN64_RUNTIME_SLOT_TABLE(HIKOBOSHI_MPNN64_ENUM)
#undef HIKOBOSHI_MPNN64_ENUM
  count,
};

/// Slot descriptor traits specialized for each hikoboshi-mpnn-d64 tensor.
template <Mpnn64SlotId Id>
struct Mpnn64SlotTraits {};

/// Typed slot descriptor for each hikoboshi-mpnn-d64 tensor.
template <Mpnn64SlotId Id>
struct Mpnn64Slot : Mpnn64SlotTraits<Id> {};

#define HIKOBOSHI_MPNN64_DEFINE_SLOT(id, tensor_name, offset, ...)       \
  template <>                                                          \
  struct Mpnn64SlotTraits<Mpnn64SlotId::id>                            \
      : TensorSlot<offset, __VA_ARGS__> {                              \
    static constexpr Mpnn64SlotId slot_id = Mpnn64SlotId::id;          \
    static constexpr std::string_view name{tensor_name};               \
  };
HIKOBOSHI_MPNN64_RUNTIME_SLOT_TABLE(HIKOBOSHI_MPNN64_DEFINE_SLOT)
#undef HIKOBOSHI_MPNN64_DEFINE_SLOT

template <Mpnn64SlotId Id>
constexpr Mpnn64SlotInfo mpnn64_slot_info() noexcept {
  using Slot = Mpnn64Slot<Id>;
  return {
      Slot::name,
      Slot::data_offset,
      Slot::shape.data(),
      Slot::rank,
      Slot::element_count,
      Slot::byte_length,
  };
}

/// Runtime slot metadata table used by validation and typed view binding.
inline constexpr Mpnn64SlotInfo kMpnn64RuntimeSlots[] = {
#define HIKOBOSHI_MPNN64_SLOT_INFO(id, name, offset, ...) \
  mpnn64_slot_info<Mpnn64SlotId::id>(),
    HIKOBOSHI_MPNN64_RUNTIME_SLOT_TABLE(HIKOBOSHI_MPNN64_SLOT_INFO)
#undef HIKOBOSHI_MPNN64_SLOT_INFO
};

inline constexpr std::size_t kMpnn64RuntimeSlotCount =
    static_cast<std::size_t>(Mpnn64SlotId::count);

static_assert(kMpnn64RuntimeSlotCount ==
                  sizeof(kMpnn64RuntimeSlots) / sizeof(kMpnn64RuntimeSlots[0]),
              "hikoboshi-mpnn-d64 slot table count mismatch");

#undef HIKOBOSHI_MPNN64_RUNTIME_SLOT_TABLE

}  // namespace hikoboshi::universal::detail

#endif  // HIKOBOSHI_UNIVERSAL_DETAIL_MPNN_D64_SLOTS_HPP
