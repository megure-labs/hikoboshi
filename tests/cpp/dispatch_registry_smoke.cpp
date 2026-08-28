#include <hikoboshi/dispatch/registry/alignment.hpp>
#include <hikoboshi/dispatch/registry/architecture.hpp>
#include <hikoboshi/dispatch/registry/capability.hpp>
#include <hikoboshi/dispatch/registry/scoring.hpp>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace hiko_dr = hikoboshi::dispatch::registry;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "dispatch_registry_smoke: %s\n", message);
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

template <typename T>
const hiko_dr::RegisteredCapabilityRecord<T>* find_capability_record(
    const hiko_u::Span<const hiko_dr::RegisteredCapabilityRecord<T>>& records,
    const T& value) {
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].value == value) {
      return &records.data[index];
    }
  }
  return nullptr;
}

const hiko_dr::RegisteredArchitectureRecord* find_record(
    hiko_u::Span<const hiko_dr::RegisteredArchitectureRecord> records,
    hiko_dr::ArchitectureKind kind) {
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].kind == kind) {
      return &records.data[index];
    }
  }
  return nullptr;
}

void test_architecture_registry_lists_mpnn_v1() {
  const hiko_u::Span<const hiko_dr::RegisteredArchitectureRecord> records =
      hiko_dr::architecture_registry();
  if (records.data == nullptr || records.size != 2) {
    fail("architecture registry must expose MPNN-64 and ESM2-8M records");
  }
  const hiko_dr::RegisteredArchitectureRecord* mpnn_record =
      find_record(records, hiko_dr::ArchitectureKind::Mpnn64);
  if (mpnn_record == nullptr) {
    fail("architecture registry must contain the MPNN-64 record");
  }
  const hiko_dr::RegisteredArchitectureRecord& record = *mpnn_record;
  if (record.architecture_id != std::string_view{"hikoboshi_mpnn_v1"} ||
      record.module_op_kind != std::string_view{"mpnn_v1_encoder"} ||
      record.scoring_op_kind !=
          std::string_view{"dot_product_similarity_v1"} ||
      record.prepared_state_kind != std::string_view{"prepared_mpnn64"} ||
      record.capability_descriptor == nullptr ||
      record.io_contract == nullptr) {
    fail("architecture record must describe hikoboshi_mpnn_v1 metadata");
  }
  if (record.capability_descriptor->input_routes.size != 3 ||
      !span_contains(record.capability_descriptor->input_routes,
                     hiko_u::PackageInputKind::StructureBackboneAtoms) ||
      !span_contains(record.capability_descriptor->input_routes,
                     hiko_u::PackageInputKind::CoordsBackbone) ||
      !span_contains(record.capability_descriptor->input_routes,
                     hiko_u::PackageInputKind::ResidueEmbeddings)) {
    fail("architecture record must list the 0.1.0 input routes");
  }
  if (record.io_contract->routes.data !=
          record.capability_descriptor->input_routes.data ||
      record.io_contract->routes.size !=
          record.capability_descriptor->input_routes.size) {
    fail("architecture IO contract must mirror capability input routes");
  }
}

void test_architecture_registry_lists_esm2_v1() {
  const hiko_u::Span<const hiko_dr::RegisteredArchitectureRecord> records =
      hiko_dr::architecture_registry();
  const hiko_dr::RegisteredArchitectureRecord* esm2_record =
      find_record(records, hiko_dr::ArchitectureKind::Esm2_8m);
  if (esm2_record == nullptr) {
    fail("architecture registry must contain the ESM2-8M record");
  }
  const hiko_dr::RegisteredArchitectureRecord& record = *esm2_record;
  if (record.architecture_id != std::string_view{"hikoboshi_esm2_v1"} ||
      record.module_op_kind != std::string_view{"plm_v1_encoder"} ||
      record.scoring_op_kind !=
          std::string_view{"dot_product_similarity_v1"} ||
      record.prepared_state_kind != std::string_view{"prepared_esm2_8m"} ||
      record.capability_descriptor == nullptr ||
      record.io_contract == nullptr) {
    fail("architecture record must describe hikoboshi_esm2_v1 metadata");
  }
  // Builder is wired up by `esm2-8m-forward`. The thunk is non-null even
  // though the embedded weights blob remains a 1-byte stub until
  // `esm2-8m-weights-package` lands; the dispatch surface must observe a
  // non-null function pointer so the closed-op-set additivity rule and
  // the validation pipeline can treat the architecture as live.
  if (record.builder == nullptr) {
    fail(
        "esm2-8m-forward must wire a non-null builder for hikoboshi_esm2_v1");
  }
  if (record.capability_descriptor->input_routes.size != 2 ||
      !span_contains(record.capability_descriptor->input_routes,
                     hiko_u::PackageInputKind::SequenceTokens) ||
      !span_contains(record.capability_descriptor->input_routes,
                     hiko_u::PackageInputKind::ResidueEmbeddings)) {
    fail("ESM2-8M record must list sequence-only input routes");
  }
  if (record.capability_descriptor->preprocessing.size != 1 ||
      record.capability_descriptor->preprocessing.data[0] !=
          hiko_u::PackagePreprocessingCapability::Tokenization) {
    fail("ESM2-8M record must require tokenization preprocessing");
  }
}

void test_architecture_lookup() {
  const hiko_dr::RegisteredArchitectureRecord* mpnn =
      hiko_dr::find_architecture("hikoboshi_mpnn_v1");
  if (mpnn == nullptr ||
      mpnn->architecture_id != std::string_view{"hikoboshi_mpnn_v1"}) {
    fail("find_architecture must resolve the registered hikoboshi_mpnn_v1 id");
  }
  const hiko_dr::RegisteredArchitectureRecord* esm2 =
      hiko_dr::find_architecture("hikoboshi_esm2_v1");
  if (esm2 == nullptr ||
      esm2->architecture_id != std::string_view{"hikoboshi_esm2_v1"}) {
    fail("find_architecture must resolve the registered hikoboshi_esm2_v1 id");
  }
  const hiko_dr::RegisteredArchitectureRecord* missing =
      hiko_dr::find_architecture("hikoboshi_gvp_v1");
  if (missing != nullptr) {
    fail("find_architecture must return nullptr for unregistered ids");
  }
}

void test_scoring_registry_lists_raw_dot_v1() {
  const hiko_u::Span<const hiko_dr::RegisteredScoringRecord> records =
      hiko_dr::scoring_registry();
  if (records.data == nullptr || records.size != 1) {
    fail("scoring registry must expose exactly one Hikoboshi 0.1.0 record");
  }
  const hiko_dr::RegisteredScoringRecord& record = records.data[0];
  if (record.kind != hiko_u::ScoreMethod::RawDotV1 ||
      record.scoring_id != std::string_view{"raw_dot_v1"} ||
      record.scoring_op_kind !=
          std::string_view{"dot_product_similarity_v1"} ||
      record.output_dtype != hiko_u::DataType::Float32) {
    fail("scoring record must describe raw_dot_v1 metadata");
  }
  if (record.inputs.size != 1 ||
      record.inputs.data[0] != hiko_u::ScoreInputKind::ResidueEmbeddings) {
    fail("scoring record must declare residue-embedding inputs");
  }
}

void test_scoring_lookup() {
  const hiko_dr::RegisteredScoringRecord* raw_dot =
      hiko_dr::find_scoring(hiko_u::ScoreMethod::RawDotV1);
  if (raw_dot == nullptr || raw_dot->kind != hiko_u::ScoreMethod::RawDotV1) {
    fail("find_scoring must resolve raw_dot_v1");
  }
  const hiko_dr::RegisteredScoringRecord* cosine =
      hiko_dr::find_scoring(hiko_u::ScoreMethod::CosineV1);
  if (cosine != nullptr) {
    fail("find_scoring must return nullptr for reserved cosine_v1");
  }
}

void test_alignment_registry_lists_hard_sw() {
  const hiko_u::Span<const hiko_dr::RegisteredAlignmentRecord> records =
      hiko_dr::alignment_registry();
  if (records.data == nullptr || records.size != 1) {
    fail("alignment registry must expose exactly one Hikoboshi 0.1.0 record");
  }
  const hiko_dr::RegisteredAlignmentRecord& record = records.data[0];
  if (record.kind != hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1 ||
      record.alignment_id != std::string_view{"hard_local_affine_sw_v1"}) {
    fail("alignment record must describe hard_local_affine_sw_v1 metadata");
  }
  if (!span_contains<std::string_view>(record.primitive_op_kinds,
                                       "smith_waterman_scalar") ||
      !span_contains<std::string_view>(record.primitive_op_kinds,
                                       "traceback_scalar")) {
    fail("alignment record must name its primitive op kinds");
  }
  if (record.gap_families_supported.size != 1 ||
      record.gap_families_supported.data[0] != hiko_u::GapModel::Affine) {
    fail("alignment record must declare affine gap support");
  }
}

void test_alignment_lookup() {
  const hiko_dr::RegisteredAlignmentRecord* hard_sw = hiko_dr::find_alignment(
      hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1);
  if (hard_sw == nullptr ||
      hard_sw->kind != hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1) {
    fail("find_alignment must resolve hard_local_affine_sw_v1");
  }
  const hiko_dr::RegisteredAlignmentRecord* soft_sw =
      hiko_dr::find_alignment(hiko_u::AlignmentAlgorithmId::SoftSwV1);
  if (soft_sw != nullptr) {
    fail("find_alignment must return nullptr for reserved soft_sw_v1");
  }
}

void test_capability_input_kinds_registry() {
  const hiko_u::Span<
      const hiko_dr::RegisteredCapabilityRecord<hiko_u::PackageInputKind>>
      records = hiko_dr::input_kinds_registry();
  if (records.data == nullptr || records.size == 0) {
    fail("input_kinds registry must be populated");
  }
  const auto* embeddings = find_capability_record(
      records, hiko_u::PackageInputKind::ResidueEmbeddings);
  if (embeddings == nullptr || !embeddings->implemented_in_0_1_0 ||
      embeddings->name != std::string_view{"residue_embeddings"}) {
    fail("input_kinds registry must mark residue_embeddings implemented");
  }
  const auto* sequence =
      find_capability_record(records, hiko_u::PackageInputKind::SequenceTokens);
  if (sequence == nullptr || !sequence->implemented_in_0_1_0) {
    fail(
        "input_kinds registry must mark sequence_tokens implemented (ESM2-8M "
        "architecture accepts the route)");
  }
  const auto* direct_score =
      find_capability_record(records, hiko_u::PackageInputKind::DirectScoreMatrix);
  if (direct_score == nullptr || direct_score->implemented_in_0_1_0) {
    fail("input_kinds registry must mark direct_score_matrix reserved");
  }
}

void test_capability_output_kinds_registry() {
  const auto records = hiko_dr::output_kinds_registry();
  const auto* embeddings = find_capability_record(
      records, hiko_u::PackageOutputKind::ResidueEmbeddings);
  if (embeddings == nullptr || !embeddings->implemented_in_0_1_0) {
    fail("output_kinds registry must mark residue_embeddings implemented");
  }
  const auto* substitution = find_capability_record(
      records, hiko_u::PackageOutputKind::SubstitutionScores);
  if (substitution == nullptr || substitution->implemented_in_0_1_0) {
    fail("output_kinds registry must mark substitution_scores reserved");
  }
}

void test_capability_preprocessing_registry() {
  const auto records = hiko_dr::preprocessing_kinds_registry();
  const auto* knn = find_capability_record(
      records, hiko_u::PackagePreprocessingCapability::CaKnn);
  if (knn == nullptr || !knn->implemented_in_0_1_0) {
    fail("preprocessing_kinds registry must mark ca_knn implemented");
  }
  const auto* tokens = find_capability_record(
      records, hiko_u::PackagePreprocessingCapability::Tokenization);
  if (tokens == nullptr || !tokens->implemented_in_0_1_0) {
    fail(
        "preprocessing_kinds registry must mark tokenization implemented "
        "(ESM2-8M ships a concrete tokenizer table)");
  }
}

void test_capability_dtypes_registry() {
  const auto records = hiko_dr::dtypes_registry();
  const auto* f32 = find_capability_record(records, hiko_u::DataType::Float32);
  if (f32 == nullptr || !f32->implemented_in_0_1_0) {
    fail("dtypes registry must mark float32 implemented");
  }
  const auto* f64 = find_capability_record(records, hiko_u::DataType::Float64);
  if (f64 == nullptr || f64->implemented_in_0_1_0) {
    fail("dtypes registry must mark float64 reserved");
  }
}

void test_capability_layouts_registry() {
  const auto records = hiko_dr::layouts_registry();
  const auto* row = find_capability_record(
      records, hiko_u::PackageTensorLayout::RowMajor);
  if (row == nullptr || !row->implemented_in_0_1_0) {
    fail("layouts registry must mark row_major implemented");
  }
}

void test_capability_devices_registry() {
  const auto records = hiko_dr::devices_registry();
  const auto* cpu = find_capability_record(records, hiko_dr::DeviceKind::Cpu);
  const auto* gpu = find_capability_record(records, hiko_dr::DeviceKind::Gpu);
  if (cpu == nullptr || !cpu->implemented_in_0_1_0) {
    fail("devices registry must mark cpu implemented");
  }
  if (gpu == nullptr || gpu->implemented_in_0_1_0) {
    fail("devices registry must mark gpu reserved");
  }
}

void test_capability_backends_registry() {
  const auto records = hiko_dr::backends_registry();
  const auto* cpu = find_capability_record(
      records, hiko_u::PackageBackendRequirement::CpuScalar);
  if (cpu == nullptr || !cpu->implemented_in_0_1_0) {
    fail("backends registry must mark cpu.scalar implemented");
  }
  const auto* cuda = find_capability_record(
      records, hiko_u::PackageBackendRequirement::GpuCuda);
  if (cuda == nullptr || cuda->implemented_in_0_1_0) {
    fail("backends registry must mark gpu.cuda reserved");
  }
}

void test_registries_are_stable_across_calls() {
  const auto first_arch = hiko_dr::architecture_registry();
  const auto second_arch = hiko_dr::architecture_registry();
  if (first_arch.data != second_arch.data ||
      first_arch.size != second_arch.size) {
    fail("architecture_registry must return a stable span across calls");
  }
  const auto first_scoring = hiko_dr::scoring_registry();
  const auto second_scoring = hiko_dr::scoring_registry();
  if (first_scoring.data != second_scoring.data ||
      first_scoring.size != second_scoring.size) {
    fail("scoring_registry must return a stable span across calls");
  }
  const auto first_alignment = hiko_dr::alignment_registry();
  const auto second_alignment = hiko_dr::alignment_registry();
  if (first_alignment.data != second_alignment.data ||
      first_alignment.size != second_alignment.size) {
    fail("alignment_registry must return a stable span across calls");
  }
  const auto first_inputs = hiko_dr::input_kinds_registry();
  const auto second_inputs = hiko_dr::input_kinds_registry();
  if (first_inputs.data != second_inputs.data ||
      first_inputs.size != second_inputs.size) {
    fail("input_kinds_registry must return a stable span across calls");
  }
}

}  // namespace

int main() {
  test_architecture_registry_lists_mpnn_v1();
  test_architecture_registry_lists_esm2_v1();
  test_architecture_lookup();
  test_scoring_registry_lists_raw_dot_v1();
  test_scoring_lookup();
  test_alignment_registry_lists_hard_sw();
  test_alignment_lookup();
  test_capability_input_kinds_registry();
  test_capability_output_kinds_registry();
  test_capability_preprocessing_registry();
  test_capability_dtypes_registry();
  test_capability_layouts_registry();
  test_capability_devices_registry();
  test_capability_backends_registry();
  test_registries_are_stable_across_calls();
  return 0;
}
