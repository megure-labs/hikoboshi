#include <hikoboshi/universal/detail/mpnn_d64_slots.hpp>
#include <hikoboshi/universal/detail/tensor_slot.hpp>
#include <hikoboshi/weights/manifest.hpp>

#include "../../cpp/weights/generated/mpnn_d64_blob.hpp"

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace hiko_s = hikoboshi::universal::detail;
namespace hiko_g = hikoboshi::weights::generated::mpnn_d64;
namespace hiko_w = hikoboshi::weights;

using WEWeightSlot = hiko_s::Mpnn64Slot<hiko_s::Mpnn64SlotId::w_e_weight>;

static_assert(std::is_same<WEWeightSlot::element_type, float>::value,
              "TensorSlot currently carries float32 slots");
static_assert(WEWeightSlot::rank == 2,
              "hikoboshi-mpnn-d64 W_e.weight slot rank mismatch");
static_assert(WEWeightSlot::shape[0] == 64 && WEWeightSlot::shape[1] == 64,
              "hikoboshi-mpnn-d64 W_e.weight slot shape mismatch");
static_assert(WEWeightSlot::element_count == 4096,
              "hikoboshi-mpnn-d64 W_e.weight slot element count mismatch");
static_assert(WEWeightSlot::byte_length == 16384,
              "hikoboshi-mpnn-d64 W_e.weight slot byte length mismatch");

std::size_t element_count_for(const std::size_t* shape,
                              std::size_t rank) noexcept {
  std::size_t count = 1;
  for (std::size_t index = 0; index < rank; ++index) {
    count *= shape[index];
  }
  return count;
}

bool same_shape(const std::size_t* lhs,
                const std::size_t* rhs,
                std::size_t rank) noexcept {
  if (lhs == nullptr || rhs == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < rank; ++index) {
    if (lhs[index] != rhs[index]) {
      return false;
    }
  }
  return true;
}

bool same_manifest_shape(const hiko_s::Mpnn64SlotInfo& slot,
                         const hiko_w::TensorManifestView& manifest_tensor) noexcept {
  return manifest_tensor.shape.size == slot.rank &&
         same_shape(slot.shape, manifest_tensor.shape.data, slot.rank);
}

int main() {
  const hiko_w::WeightManifestView& manifest = hiko_w::default_mpnn_d64_manifest();
  if (hiko_s::kMpnn64RuntimeSlotCount != hiko_g::kRuntimeTensorCount ||
      manifest.tensors.size != hiko_s::kMpnn64RuntimeSlotCount) {
    return 1;
  }

  for (std::size_t index = 0; index < hiko_s::kMpnn64RuntimeSlotCount; ++index) {
    const hiko_s::Mpnn64SlotInfo& slot = hiko_s::kMpnn64RuntimeSlots[index];
    const hiko_g::TensorBlobInfo& runtime_tensor = hiko_g::kRuntimeTensors[index];
    const hiko_w::TensorManifestView& manifest_tensor =
        manifest.tensors.data[index];

    if (slot.name != runtime_tensor.name ||
        slot.data_offset != runtime_tensor.data_offset ||
        slot.rank != runtime_tensor.rank ||
        !same_shape(slot.shape, runtime_tensor.shape, slot.rank) ||
        slot.element_count != runtime_tensor.byte_length / sizeof(float) ||
        slot.byte_length != runtime_tensor.byte_length ||
        slot.data_offset + slot.byte_length > hiko_g::kSafetensorsDataLength ||
        runtime_tensor.dtype != std::string_view{"float32"}) {
      return 2;
    }

    if (manifest_tensor.name != slot.name ||
        manifest_tensor.dtype != std::string_view{"float32"} ||
        !same_manifest_shape(slot, manifest_tensor) ||
        slot.byte_length !=
            element_count_for(manifest_tensor.shape.data,
                              manifest_tensor.shape.size) *
                sizeof(float)) {
      return 3;
    }
  }

  return 0;
}
