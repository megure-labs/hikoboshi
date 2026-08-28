#include <hikoboshi/dispatch/dispatch_table.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>

#include <hikoboshi/universal/backend.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor_role.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace hiko_dr = hikoboshi::dispatch::registry;
namespace hiko_d = hikoboshi::dispatch;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "primitive_op_registry_tests: %s\n", message);
  std::exit(1);
}

template <typename T>
bool span_contains(const hiko_u::Span<const T>& span, const T& value) {
  for (std::size_t index = 0; index < span.size; ++index) {
    if (span.data[index] == value) {
      return true;
    }
  }
  return false;
}

struct ExpectedOp {
  std::string_view op_id;
  std::string_view family;
  hiko_dr::PrimitiveOpFamily family_enum;
};

constexpr ExpectedOp kExpectedOps[] = {
    {"hikoboshi.knn.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.rbf.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.gather.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.layer_norm.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.reduce_sum_rows.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.reduce_mean_rows.v1", "compute",
     hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.softmax.row_wise.v1", "compute",
     hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.log_softmax.row_wise.v1", "compute",
     hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.bias_add.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.gelu.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.axpy.v1", "compute", hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.atom_pair_distance.v1", "compute",
     hiko_dr::PrimitiveOpFamily::Compute},
    {"hikoboshi.gemm.nn.v1", "linalg", hiko_dr::PrimitiveOpFamily::Linalg},
    {"hikoboshi.gemm.nt.v1", "linalg", hiko_dr::PrimitiveOpFamily::Linalg},
    {"hikoboshi.smith_waterman.v1", "alignment",
     hiko_dr::PrimitiveOpFamily::Alignment},
    {"hikoboshi.traceback.v1", "alignment", hiko_dr::PrimitiveOpFamily::Alignment},
};

constexpr std::size_t kExpectedOpCount =
    sizeof(kExpectedOps) / sizeof(kExpectedOps[0]);

void test_registry_has_expected_count() {
  const hiko_u::Span<const hiko_dr::RegisteredPrimitiveOpRecord> records =
      hiko_dr::primitive_op_registry();
  if (records.data == nullptr) {
    fail("primitive_op_registry must return non-null data");
  }
  if (records.size != kExpectedOpCount) {
    fail("primitive_op_registry must list every closed Hikoboshi 0.1.0 op");
  }
}

void test_registry_lists_each_expected_op() {
  for (std::size_t i = 0; i < kExpectedOpCount; ++i) {
    const ExpectedOp& expected = kExpectedOps[i];
    const hiko_dr::RegisteredPrimitiveOpRecord* record =
        hiko_dr::find_primitive_op(expected.op_id);
    if (record == nullptr) {
      fail("primitive_op_registry must include every expected op id");
    }
    if (record->identity.op_id != expected.op_id ||
        record->identity.op_family != expected.family ||
        record->identity.op_version != std::string_view{"v1"} ||
        record->family != expected.family_enum) {
      fail("primitive op identity must match the expected family and version");
    }
    if (record->dispatch_entry == nullptr) {
      fail("primitive op must register a non-null dispatch entry");
    }
  }
}

void test_capabilities_declare_scalar_strict() {
  // Every registered op must support the cpu.scalar backend and must
  // expose strict parity as one of its supported modes so callers can
  // always fall back to bit-identity numerics. Individual ops may
  // declare a non-strict `default_parity_mode` (softmax defaults to
  // Fast); the default still has to appear in the supported list.
  const hiko_u::Span<const hiko_dr::RegisteredPrimitiveOpRecord> records =
      hiko_dr::primitive_op_registry();
  for (std::size_t i = 0; i < records.size; ++i) {
    const hiko_dr::PrimitiveOpCapabilities& cap = records.data[i].capabilities;
    if (!span_contains(cap.supported_backends,
                       hiko_u::PackageBackendRequirement::CpuScalar)) {
      fail("every Hikoboshi 0.1.0 primitive op must support cpu.scalar");
    }
    if (!span_contains<hiko_dr::ParityMode>(cap.supported_parity_modes,
                                         hiko_dr::ParityMode::Strict)) {
      fail(
          "every Hikoboshi 0.1.0 primitive op must support strict parity mode");
    }
    if (!span_contains<hiko_dr::ParityMode>(cap.supported_parity_modes,
                                         cap.default_parity_mode)) {
      fail(
          "primitive op default_parity_mode must appear in "
          "supported_parity_modes");
    }
  }
}

void test_find_returns_null_for_unknown_id() {
  if (hiko_dr::find_primitive_op("hikoboshi.softmax.v1") != nullptr) {
    fail("find_primitive_op must return nullptr for unknown ids");
  }
  if (hiko_dr::find_primitive_op("") != nullptr) {
    fail("find_primitive_op must return nullptr for empty op id");
  }
}

void test_registry_is_stable_across_calls() {
  const auto first = hiko_dr::primitive_op_registry();
  const auto second = hiko_dr::primitive_op_registry();
  if (first.data != second.data || first.size != second.size) {
    fail("primitive_op_registry must return a stable span across calls");
  }
}

void test_signatures_are_populated() {
  const hiko_dr::RegisteredPrimitiveOpRecord* knn =
      hiko_dr::find_primitive_op("hikoboshi.knn.v1");
  if (knn == nullptr) {
    fail("knn record must be present");
  }
  if (knn->signature.outputs.size != 2 ||
      !span_contains(knn->signature.outputs,
                     hiko_u::TensorRole::NeighborIndices) ||
      !span_contains(knn->signature.outputs,
                     hiko_u::TensorRole::SquaredDistances)) {
    fail("knn op must declare neighbor_indices and squared_distances outputs");
  }
  if (!span_contains(knn->signature.parameters,
                     hiko_u::ParameterRole::NeighborCount)) {
    fail("knn op must declare a neighbor_count parameter role");
  }

  const hiko_dr::RegisteredPrimitiveOpRecord* gemm_nt =
      hiko_dr::find_primitive_op("hikoboshi.gemm.nt.v1");
  if (gemm_nt == nullptr) {
    fail("gemm_nt record must be present");
  }
  if (!span_contains(gemm_nt->signature.inputs,
                     hiko_u::TensorRole::EncoderWeight)) {
    fail("gemm_nt op must declare encoder_weight as an input role");
  }

  const hiko_dr::RegisteredPrimitiveOpRecord* layer_norm =
      hiko_dr::find_primitive_op("hikoboshi.layer_norm.v1");
  if (layer_norm == nullptr) {
    fail("layer_norm record must be present");
  }
  if (!span_contains(layer_norm->signature.inputs,
                     hiko_u::TensorRole::EncoderNormGamma) ||
      !span_contains(layer_norm->signature.inputs,
                     hiko_u::TensorRole::EncoderNormBeta)) {
    fail(
        "layer_norm op must declare encoder_norm_gamma and encoder_norm_beta "
        "input roles");
  }
}

void test_dispatch_table_populated_from_registry() {
  const hiko_d::DispatchTable& table = hiko_d::scalar_dispatch_table();
  if (table.backend != hiko_u::Backend::Scalar) {
    fail("scalar dispatch table must report Backend::Scalar");
  }
  if (table.knn == nullptr || table.rbf == nullptr || table.gather == nullptr ||
      table.layer_norm == nullptr || table.reduce_sum_rows == nullptr ||
      table.reduce_mean_rows == nullptr || table.softmax == nullptr ||
      table.log_softmax == nullptr || table.gemm_nn == nullptr ||
      table.gemm_nt == nullptr || table.smith_waterman == nullptr ||
      table.traceback == nullptr) {
    fail("scalar dispatch table must populate every typed slot");
  }

  const hiko_dr::RegisteredPrimitiveOpRecord* knn =
      hiko_dr::find_primitive_op("hikoboshi.knn.v1");
  const hiko_dr::RegisteredPrimitiveOpRecord* log_softmax =
      hiko_dr::find_primitive_op("hikoboshi.log_softmax.row_wise.v1");
  const hiko_dr::RegisteredPrimitiveOpRecord* gemm_nt =
      hiko_dr::find_primitive_op("hikoboshi.gemm.nt.v1");
  if (knn == nullptr || log_softmax == nullptr || gemm_nt == nullptr) {
    fail("expected primitive records must be available");
  }
  if (reinterpret_cast<const void*>(table.knn) != knn->dispatch_entry) {
    fail("scalar dispatch table knn slot must point at the registry entry");
  }
  if (reinterpret_cast<const void*>(table.log_softmax) !=
      log_softmax->dispatch_entry) {
    fail(
        "scalar dispatch table log_softmax slot must point at the registry "
        "entry");
  }
  if (reinterpret_cast<const void*>(table.gemm_nt) !=
      gemm_nt->dispatch_entry) {
    fail(
        "scalar dispatch table gemm_nt slot must point at the registry entry");
  }
}

}  // namespace

int main() {
  test_registry_has_expected_count();
  test_registry_lists_each_expected_op();
  test_capabilities_declare_scalar_strict();
  test_find_returns_null_for_unknown_id();
  test_registry_is_stable_across_calls();
  test_signatures_are_populated();
  test_dispatch_table_populated_from_registry();
  return 0;
}
