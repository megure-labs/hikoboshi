#include <hikoboshi/dispatch/registry/architecture.hpp>
#include <hikoboshi/dispatch/registry/module_op.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor_role.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace hiko_dr = hikoboshi::dispatch::registry;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "module_op_registry_tests: %s\n", message);
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

struct ExpectedModuleOp {
  std::string_view op_id;
  std::string_view family;
  hiko_dr::ModuleOpFamily family_enum;
};

constexpr ExpectedModuleOp kExpectedModuleOps[] = {
    {"hikoboshi.mpnn.v1.encoder", "encoder", hiko_dr::ModuleOpFamily::Encoder},
    {"hikoboshi.similarity.dot_product.v1", "similarity",
     hiko_dr::ModuleOpFamily::Similarity},
    {"hikoboshi.mpnn.v1.message_layer", "layer", hiko_dr::ModuleOpFamily::Layer},
    {"hikoboshi.mpnn.v1.ffn_layer", "layer", hiko_dr::ModuleOpFamily::Layer},
    {"hikoboshi.alignment.hard_sw_wrapper.v1", "alignment_wrapper",
     hiko_dr::ModuleOpFamily::AlignmentWrapper},
    // POST010 attention-compound-module registration. The generic attention
    // compound is the first module-op whose default parity mode is `Fast`;
    // the strict-vs-fast invariant test below exempts it explicitly. The op
    // id was re-keyed from `hikoboshi.transformer.attention.v1` to
    // `hikoboshi.attention.v1` by the compound-module-architecture-agnostic-
    // interface packet because attention is composable by any attention-
    // based architecture; the textual family spells "attention" rather than
    // "layer" for the same reason.
    {"hikoboshi.attention.v1", "attention", hiko_dr::ModuleOpFamily::Layer},
    // Generic FFN template family. CUTLASS-style closed-tag-axis
    // specialization surface; defaults to Fast parity for the same
    // stacked-GEMM-hot-path reason as attention. The textual family spells
    // "feedforward" while the enum bucket stays `Layer`.
    {"hikoboshi.ffn_layer.v1", "feedforward", hiko_dr::ModuleOpFamily::Layer},
    // POST010 esm2-8m-forward registration. ESM2-8M is the second registered
    // encoder architecture; its compound op composes the registered
    // attention + FFN compounds plus the scalar LayerNorm primitive. Defaults
    // to Fast parity for the same stacked-GEMM-hot-path reason as the
    // attention and FFN compounds.
    {"hikoboshi.esm2.v1.encoder", "encoder", hiko_dr::ModuleOpFamily::Encoder},
};

constexpr std::size_t kExpectedModuleOpCount =
    sizeof(kExpectedModuleOps) / sizeof(kExpectedModuleOps[0]);

void test_registry_has_expected_count() {
  const hiko_u::Span<const hiko_dr::RegisteredModuleOpRecord> records =
      hiko_dr::module_op_registry();
  if (records.data == nullptr) {
    fail("module_op_registry must return non-null data");
  }
  if (records.size != kExpectedModuleOpCount) {
    fail(
        "module_op_registry must list every Hikoboshi 0.1.0 compound module "
        "op");
  }
}

void test_registry_lists_each_expected_op() {
  for (std::size_t i = 0; i < kExpectedModuleOpCount; ++i) {
    const ExpectedModuleOp& expected = kExpectedModuleOps[i];
    const hiko_dr::RegisteredModuleOpRecord* record =
        hiko_dr::find_module_op(expected.op_id);
    if (record == nullptr) {
      fail("module_op_registry must include every expected op id");
    }
    if (record->identity.op_id != expected.op_id ||
        record->identity.op_family != expected.family ||
        record->identity.op_version != std::string_view{"v1"} ||
        record->family != expected.family_enum) {
      fail("module-op identity must match the expected family and version");
    }
    if (record->dispatch_entry == nullptr) {
      fail("module-op must register a non-null dispatch entry");
    }
  }
}

void test_capabilities_declare_scalar_strict() {
  const hiko_u::Span<const hiko_dr::RegisteredModuleOpRecord> records =
      hiko_dr::module_op_registry();
  for (std::size_t i = 0; i < records.size; ++i) {
    const hiko_dr::RegisteredModuleOpRecord& record = records.data[i];
    const hiko_dr::ModuleOpCapabilities& cap = record.capabilities;
    if (!span_contains(cap.supported_backends,
                       hiko_u::PackageBackendRequirement::CpuScalar)) {
      fail("every Hikoboshi 0.1.0 module-op must support cpu.scalar");
    }
    if (!span_contains<hiko_dr::ParityMode>(cap.supported_parity_modes,
                                         hiko_dr::ParityMode::Strict)) {
      fail("every Hikoboshi 0.1.0 module-op must support strict parity mode");
    }
    // The Hikoboshi 0.1.0 closed module-op set defaults to strict parity. The
    // POST010 generic compounds (attention block, FFN template family) and
    // the ESM2-8M encoder default to fast because each is a stacked-GEMM
    // hot path that benefits from the fast kernel out of the box; strict
    // remains a supported mode for parity goldens. Exempt those records
    // from the strict-default invariant while keeping the invariant in
    // force for every other op.
    const std::string_view op_id = record.identity.op_id;
    if (op_id != std::string_view{"hikoboshi.attention.v1"} &&
        op_id != std::string_view{"hikoboshi.ffn_layer.v1"} &&
        op_id != std::string_view{"hikoboshi.esm2.v1.encoder"} &&
        cap.default_parity_mode != hiko_dr::ParityMode::Strict) {
      fail("Hikoboshi 0.1.0 default parity mode must be strict");
    }
  }
}

void test_find_returns_null_for_unknown_id() {
  if (hiko_dr::find_module_op("hikoboshi.transformer.attention.v1") != nullptr) {
    fail(
        "find_module_op must return nullptr for the pre-rekey attention op "
        "id (re-keyed to hikoboshi.attention.v1)");
  }
  if (hiko_dr::find_module_op("hikoboshi.this_module_does_not_exist.v1") !=
      nullptr) {
    fail("find_module_op must return nullptr for unknown ids");
  }
  if (hiko_dr::find_module_op("") != nullptr) {
    fail("find_module_op must return nullptr for empty op id");
  }
}

void test_registry_is_stable_across_calls() {
  const auto first = hiko_dr::module_op_registry();
  const auto second = hiko_dr::module_op_registry();
  if (first.data != second.data || first.size != second.size) {
    fail("module_op_registry must return a stable span across calls");
  }
}

void test_required_primitive_ids_resolve_in_primitive_registry() {
  const hiko_u::Span<const hiko_dr::RegisteredModuleOpRecord> records =
      hiko_dr::module_op_registry();
  for (std::size_t i = 0; i < records.size; ++i) {
    const hiko_dr::RegisteredModuleOpRecord& record = records.data[i];
    const hiko_u::Span<const std::string_view> required =
        record.signature.required_primitive_op_ids;
    if (required.size == 0) {
      fail("every module-op must declare at least one required primitive op");
    }
    for (std::size_t r = 0; r < required.size; ++r) {
      const std::string_view required_id = required.data[r];
      const hiko_dr::RegisteredPrimitiveOpRecord* primitive =
          hiko_dr::find_primitive_op(required_id);
      if (primitive == nullptr) {
        fail(
            "module-op required_primitive_op_ids entry must resolve in "
            "primitive_op_registry()");
      }
    }
  }
}

void test_live_validation_passes() {
  const hiko_dr::ModuleOpValidationReport& report =
      hiko_dr::module_op_registry_validation();
  if (!report.ok) {
    fail("live module-op cross-reference validation must report ok");
  }
  if (report.diagnostics.size != 0) {
    fail(
        "live module-op cross-reference validation must report zero "
        "diagnostics");
  }
}

void test_live_validation_is_cached() {
  const hiko_dr::ModuleOpValidationReport& first =
      hiko_dr::module_op_registry_validation();
  const hiko_dr::ModuleOpValidationReport& second =
      hiko_dr::module_op_registry_validation();
  if (&first != &second) {
    fail("module_op_registry_validation must return a cached reference");
  }
}

void test_injected_missing_primitive_fires_structured_diagnostic() {
  // Build a fake module-op record that declares a primitive id which is
  // intentionally not in the live primitive_op_registry(). The
  // validate_module_op_records path must emit a structured diagnostic.
  static constexpr std::string_view kFakeRequired[] = {
      "hikoboshi.this_primitive_does_not_exist.v1",
  };
  static constexpr hiko_u::TensorRole kFakeInputs[] = {
      hiko_u::TensorRole::ResidueEmbeddings,
  };
  static constexpr hiko_u::TensorRole kFakeOutputs[] = {
      hiko_u::TensorRole::ResidueEmbeddings,
  };
  static constexpr hiko_u::PackageBackendRequirement kFakeBackends[] = {
      hiko_u::PackageBackendRequirement::CpuScalar,
  };
  static constexpr hiko_dr::ParityMode kFakeParity[] = {hiko_dr::ParityMode::Strict};
  const hiko_dr::RegisteredModuleOpRecord fake_record{
      {"hikoboshi.fake_module.v1", "encoder", "v1"},
      hiko_dr::ModuleOpFamily::Encoder,
      {{kFakeInputs, 1}, {kFakeOutputs, 1}, {kFakeRequired, 1}},
      {{kFakeBackends, 1}, {kFakeParity, 1}, hiko_dr::ParityMode::Strict},
      reinterpret_cast<const void*>(&fail),
  };
  const hiko_dr::RegisteredModuleOpRecord records_array[] = {fake_record};
  hiko_dr::ModuleOpValidationDiagnostic buffer[4]{};
  const hiko_dr::ModuleOpValidationReport report =
      hiko_dr::validate_module_op_records({records_array, 1}, buffer, 4);
  if (report.ok) {
    fail(
        "injected missing primitive must cause validate_module_op_records to "
        "report ok=false");
  }
  if (report.diagnostics.size != 1) {
    fail(
        "injected missing primitive must produce exactly one diagnostic for "
        "the single missing id");
  }
  const hiko_dr::ModuleOpValidationDiagnostic& diag = report.diagnostics.data[0];
  if (diag.severity != hiko_dr::ModuleOpValidationSeverity::Error) {
    fail("injected missing primitive diagnostic must be Error severity");
  }
  if (diag.code != std::string_view{"module_op_required_primitive_missing"}) {
    fail(
        "injected missing primitive diagnostic must use the structured "
        "module_op_required_primitive_missing code");
  }
  if (diag.module_op_id != std::string_view{"hikoboshi.fake_module.v1"}) {
    fail("diagnostic must identify the offending module-op id");
  }
  if (diag.missing_primitive_op_id !=
      std::string_view{"hikoboshi.this_primitive_does_not_exist.v1"}) {
    fail("diagnostic must identify the unresolved primitive-op id");
  }
  if (diag.message.empty()) {
    fail("diagnostic message must be populated");
  }
}

void test_validate_module_op_records_passes_for_real_records() {
  // Reuse the live registry records through the explicit validation path to
  // double-check that the test-seam entry point agrees with the cached
  // singleton.
  const hiko_u::Span<const hiko_dr::RegisteredModuleOpRecord> records =
      hiko_dr::module_op_registry();
  hiko_dr::ModuleOpValidationDiagnostic buffer[96]{};
  const hiko_dr::ModuleOpValidationReport report =
      hiko_dr::validate_module_op_records(records, buffer, 96);
  if (!report.ok) {
    fail(
        "validate_module_op_records on the live registry must agree with the "
        "cached singleton and report ok");
  }
  if (report.diagnostics.size != 0) {
    fail(
        "validate_module_op_records on the live registry must emit zero "
        "diagnostics");
  }
}

void test_architecture_required_module_op_ids_resolve() {
  const hiko_dr::RegisteredArchitectureRecord* mpnn =
      hiko_dr::find_architecture("hikoboshi_mpnn_v1");
  if (mpnn == nullptr) {
    fail("MPNN-64 architecture record must be present in the registry");
  }
  if (mpnn->required_module_op_ids.size == 0) {
    fail(
        "MPNN-64 architecture record must list its composing module-op ids "
        "in required_module_op_ids");
  }
  for (std::size_t i = 0; i < mpnn->required_module_op_ids.size; ++i) {
    const std::string_view id = mpnn->required_module_op_ids.data[i];
    if (hiko_dr::find_module_op(id) == nullptr) {
      fail(
          "every architecture required_module_op_ids entry must resolve in "
          "module_op_registry()");
    }
  }
  // The MPNN-64 architecture composes the encoder and the dot-product
  // similarity module; assert those entries are present specifically.
  if (!span_contains<std::string_view>(mpnn->required_module_op_ids,
                                       "hikoboshi.mpnn.v1.encoder") ||
      !span_contains<std::string_view>(
          mpnn->required_module_op_ids,
          "hikoboshi.similarity.dot_product.v1")) {
    fail(
        "MPNN-64 required_module_op_ids must list the encoder and the "
        "similarity module-op");
  }
}

void test_signatures_are_populated() {
  const hiko_dr::RegisteredModuleOpRecord* encoder =
      hiko_dr::find_module_op("hikoboshi.mpnn.v1.encoder");
  if (encoder == nullptr) {
    fail("encoder record must be present");
  }
  if (!span_contains(encoder->signature.outputs,
                     hiko_u::TensorRole::ResidueEmbeddings)) {
    fail("encoder module-op must declare residue_embeddings as an output role");
  }
  if (!span_contains<std::string_view>(
          encoder->signature.required_primitive_op_ids,
          "hikoboshi.gemm.nt.v1") ||
      !span_contains<std::string_view>(
          encoder->signature.required_primitive_op_ids, "hikoboshi.knn.v1")) {
    fail(
        "encoder module-op must declare gemm_nt and knn as required "
        "primitives");
  }

  const hiko_dr::RegisteredModuleOpRecord* similarity =
      hiko_dr::find_module_op("hikoboshi.similarity.dot_product.v1");
  if (similarity == nullptr) {
    fail("similarity record must be present");
  }
  if (!span_contains(similarity->signature.outputs,
                     hiko_u::TensorRole::ScoreMatrix)) {
    fail("similarity module-op must declare score_matrix as an output role");
  }

  const hiko_dr::RegisteredModuleOpRecord* hard_sw =
      hiko_dr::find_module_op("hikoboshi.alignment.hard_sw_wrapper.v1");
  if (hard_sw == nullptr) {
    fail("hard_sw_wrapper record must be present");
  }
  if (!span_contains(hard_sw->signature.outputs,
                     hiko_u::TensorRole::AlignmentPath)) {
    fail(
        "hard_sw_wrapper module-op must declare alignment_path as an output "
        "role");
  }
  if (!span_contains<std::string_view>(
          hard_sw->signature.required_primitive_op_ids,
          "hikoboshi.smith_waterman.v1") ||
      !span_contains<std::string_view>(
          hard_sw->signature.required_primitive_op_ids,
          "hikoboshi.traceback.v1")) {
    fail(
        "hard_sw_wrapper module-op must declare smith_waterman and traceback "
        "as required primitives");
  }
}

}  // namespace

int main() {
  test_registry_has_expected_count();
  test_registry_lists_each_expected_op();
  test_capabilities_declare_scalar_strict();
  test_find_returns_null_for_unknown_id();
  test_registry_is_stable_across_calls();
  test_required_primitive_ids_resolve_in_primitive_registry();
  test_live_validation_passes();
  test_live_validation_is_cached();
  test_injected_missing_primitive_fires_structured_diagnostic();
  test_validate_module_op_records_passes_for_real_records();
  test_architecture_required_module_op_ids_resolve();
  test_signatures_are_populated();
  return 0;
}
