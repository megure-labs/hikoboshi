#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/api/requests.hpp>
#include <hikoboshi/api/results.hpp>
#include <hikoboshi/api/types.hpp>
#include <hikoboshi/api/version.hpp>
#include <hikoboshi/universal/planner.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hiko = hikoboshi::api;
namespace hiko_u = hikoboshi::universal;

static_assert(std::is_standard_layout<hiko::AlignmentStep>::value,
              "AlignmentStep must stay a plain result record");
static_assert(std::is_same<hiko::AlignmentStep, hiko_u::AlignmentStep>::value,
              "api keeps one-release AlignmentStep re-export");
static_assert(std::is_standard_layout<hiko::CoordsInputView>::value,
              "CoordsInputView must stay a plain request view");
static_assert(std::is_standard_layout<hiko::AllVsAllStructureRequest>::value,
              "AllVsAllStructureRequest must stay a plain request view");
static_assert(std::is_standard_layout<hiko::PairwiseResultRecord>::value,
              "PairwiseResultRecord must stay a plain result record");
static_assert(std::is_standard_layout<hiko_u::PlannerPolicy>::value,
              "PlannerPolicy must stay a low-behavior public policy record");
static_assert(std::is_same<decltype(std::declval<hiko::EngineConfig>().planner_policy),
                           const hiko_u::PlannerPolicy*>::value,
              "EngineConfig must carry planner policy as an immutable pointer");
static_assert(std::is_same<hiko_u::Status (hiko::PairwiseResultSink::*)(
                               const hiko::PairwiseResultRecord&),
                           decltype(&hiko::PairwiseResultSink::receive)>::value,
              "PairwiseResultSink must return structured status");
static_assert(hiko::kDefaultGapOpen == -1.4F,
              "API default gap open must match the hard-SW charter");
static_assert(hiko::kDefaultGapExtension == -0.15F,
              "API default gap extension must match the hard-SW charter");

template <typename T, typename = void>
struct HasSubstitutionMatrixMethod : std::false_type {};

template <typename T>
struct HasSubstitutionMatrixMethod<
    T,
    std::void_t<decltype(&T::substitution_matrix)>> : std::true_type {};

template <typename T, typename = void>
struct HasSubstitutionLookupMethod : std::false_type {};

template <typename T>
struct HasSubstitutionLookupMethod<
    T,
    std::void_t<decltype(&T::substitution_lookup)>> : std::true_type {};

template <typename T, typename = void>
struct HasSequencePairwiseMethod : std::false_type {};

template <typename T>
struct HasSequencePairwiseMethod<
    T,
    std::void_t<decltype(&T::sequence_pairwise)>> : std::true_type {};

template <typename T, typename = void>
struct HasFlatAutoBackendField : std::false_type {};

template <typename T>
struct HasFlatAutoBackendField<
    T,
    std::void_t<decltype(std::declval<T>().auto_backend)>> : std::true_type {};

template <typename T, typename = void>
struct HasFlatScalarBackendField : std::false_type {};

template <typename T>
struct HasFlatScalarBackendField<
    T,
    std::void_t<decltype(std::declval<T>().scalar_backend)>> : std::true_type {};

template <typename T, typename = void>
struct HasFlatGpuBackendField : std::false_type {};

template <typename T>
struct HasFlatGpuBackendField<
    T,
    std::void_t<decltype(std::declval<T>().gpu_backend)>> : std::true_type {};

static_assert(!HasSubstitutionMatrixMethod<hiko::Engine>::value,
              "Hikoboshi 0.1.0 must not expose Engine::substitution_matrix");
static_assert(!HasSubstitutionLookupMethod<hiko::Engine>::value,
              "Hikoboshi 0.1.0 must not expose Engine::substitution_lookup");
static_assert(!HasSequencePairwiseMethod<hiko::Engine>::value,
              "Hikoboshi 0.1.0 must not expose Engine::sequence_pairwise");
static_assert(!HasFlatAutoBackendField<hiko::BackendCapabilities>::value,
              "BackendCapabilities must not expose old flat auto_backend");
static_assert(!HasFlatScalarBackendField<hiko::BackendCapabilities>::value,
              "BackendCapabilities must not expose old flat scalar_backend");
static_assert(!HasFlatGpuBackendField<hiko::BackendCapabilities>::value,
              "BackendCapabilities must not expose old flat gpu_backend");

class RecordingSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    last_query_index = record.query_index;
    last_target_index = record.target_index;
    return {hiko_u::StatusCode::Ok, nullptr};
  }

  std::size_t last_query_index = 0;
  std::size_t last_target_index = 0;
};

int main() {
  constexpr std::size_t residue_count = 2;
  constexpr std::size_t embedding_dimension = 64;

  const float coordinates[residue_count * hiko_u::kCanonicalAtomCount *
                          hiko_u::kCoordinateAxisCount] = {};
  const hiko_u::AtomSource atom_sources[residue_count * hiko_u::kCanonicalAtomCount] = {};
  const char residue_codes[residue_count] = {'A', 'G'};
  const hiko_u::ResidueMetadataView residues[residue_count] = {
      {'A', "ALA", "A", "1", 1, 1, '\0', "query", 0, "query.pdb", 10},
      {'G', "GLY", "A", "1", 1, 2, '\0', "query", 1, "query.pdb", 20},
  };
  const hiko_u::ChainBreakView chain_breaks[1] = {{0}};

  const hiko_u::StructureView structure{
      residue_count,
      {coordinates, residue_count * hiko_u::kCanonicalAtomCount *
                        hiko_u::kCoordinateAxisCount},
      {atom_sources, residue_count * hiko_u::kCanonicalAtomCount},
      {residue_codes, residue_count},
      {residues, residue_count},
      "query",
      "query.pdb",
      {chain_breaks, 1},
  };
  const hiko_u::StructureView structures[2] = {structure, structure};

  const hiko::CoordsInputView coords{
      residue_count,
      {coordinates, residue_count * hiko_u::kCanonicalAtomCount *
                        hiko_u::kCoordinateAxisCount},
      {atom_sources, residue_count * hiko_u::kCanonicalAtomCount},
      {residue_codes, residue_count},
      {residues, residue_count},
  };
  const hiko::CoordsInputView coord_inputs[2] = {coords, coords};
  const hiko_u::Span<const hiko::CoordsInputView> coord_input_span{coord_inputs, 2};

  const float embedding_values[residue_count * embedding_dimension] = {};
  const hiko_u::EmbeddingView embedding{
      residue_count,
      embedding_dimension,
      {embedding_values, residue_count * embedding_dimension},
      {residue_codes, residue_count},
      {residues, residue_count},
  };
  const hiko_u::EmbeddingView embeddings[2] = {embedding, embedding};
  const hiko_u::Span<const hiko_u::EmbeddingView> embedding_span{embeddings, 2};
  const hiko_u::Status constructed_ok = hiko_u::ok_status();

  const hiko::EngineConfig config{
      {nullptr, nullptr},
      {hiko_u::Backend::Auto, 0},
  };
  const hiko::Engine engine(config);
  const hiko_u::PlannerPolicy& planner_policy =
      hiko_u::scalar_default_planner_policy(engine.config().planner_policy);
  RecordingSink sink;

  const auto encoded_structure = engine.encode(hiko::EncodeStructureRequest{structure});
  const auto encoded_coords = engine.encode(hiko::EncodeCoordsRequest{coords});
  const auto pairwise_structure =
      engine.pairwise(hiko::PairwiseStructureRequest{structure, structure});
  const auto pairwise_coords =
      engine.pairwise(hiko::PairwiseCoordsRequest{coords, coords});
  const auto pairwise_embedding =
      engine.pairwise(hiko::PairwiseEmbeddingRequest{embedding, embedding});
  const auto all_structure =
      engine.all_vs_all(hiko::AllVsAllStructureRequest{{structures, 2}}, sink);
  const auto all_coords =
      engine.all_vs_all(hiko::AllVsAllCoordsRequest{coord_input_span}, sink);
  const auto all_embedding =
      engine.all_vs_all(hiko::AllVsAllEmbeddingRequest{embedding_span}, sink);
  const auto collected_structure =
      engine.collect_all_vs_all(hiko::AllVsAllStructureRequest{{structures, 2}});
  const auto collected_coords =
      engine.collect_all_vs_all(hiko::AllVsAllCoordsRequest{coord_input_span});
  const auto collected_embedding =
      engine.collect_all_vs_all(hiko::AllVsAllEmbeddingRequest{embedding_span});
  const auto collected_free =
      hiko::collect_all_vs_all(engine,
                              hiko::AllVsAllEmbeddingRequest{embedding_span});

  const hiko::VersionInfo version = hiko::version_info();
  const hiko::BackendCapabilities capabilities = engine.backend_capabilities();

  return encoded_structure.status.code == hiko_u::StatusCode::FailedPrecondition &&
                 encoded_coords.status.code == hiko_u::StatusCode::FailedPrecondition &&
                 pairwise_structure.status.code == hiko_u::StatusCode::FailedPrecondition &&
                 pairwise_coords.status.code == hiko_u::StatusCode::FailedPrecondition &&
                 pairwise_embedding.status.code == hiko_u::StatusCode::Ok &&
                 all_structure.code == hiko_u::StatusCode::FailedPrecondition &&
                 all_coords.code == hiko_u::StatusCode::FailedPrecondition &&
                 all_embedding.code == hiko_u::StatusCode::Ok &&
                 collected_structure.status.code == hiko_u::StatusCode::FailedPrecondition &&
                 collected_coords.status.code == hiko_u::StatusCode::FailedPrecondition &&
                 collected_embedding.status.code == hiko_u::StatusCode::Ok &&
                 collected_embedding.value.records.size() == 1 &&
                 collected_free.status.code == hiko_u::StatusCode::Ok &&
                 collected_free.value.records.size() == 1 &&
                 engine.config().execution.backend == hiko_u::Backend::Auto &&
                 engine.config().planner_policy == nullptr &&
                 planner_policy.default_backend == hiko_u::Backend::Scalar &&
                 planner_policy.fallback_order.size == 1 &&
                 !planner_policy.fallback_order.empty() &&
                 planner_policy.fallback_order.front() == hiko_u::Backend::Scalar &&
                 embedding_span[1].dimension == embedding_dimension &&
                 constructed_ok.ok() &&
                 version.version.minor == 1 &&
                 capabilities.cpu.scalar.compiled &&
                 capabilities.cpu.scalar.runtime_available &&
                 !capabilities.cpu.avx2.compiled &&
                 !capabilities.gpu.cuda.availability.compiled &&
                 capabilities.gpu.cuda.devices.size == 0 &&
                 capabilities.pipeline.symmetric_all_vs_all
             ? 0
             : 1;
}
