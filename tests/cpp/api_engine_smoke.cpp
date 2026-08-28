#include <hikoboshi/algorithms/all_vs_all.hpp>
#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_g = hikoboshi::algorithms;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "api_engine_smoke: %s\n", message);
  std::exit(1);
}

bool nearly_equal(double a, double b, double tolerance = 1.0e-6) {
  return std::fabs(a - b) <= tolerance;
}

bool contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

std::string_view detail(hiko_u::Status status) {
  return status.detail == nullptr ? std::string_view{} : std::string_view{status.detail};
}

constexpr hiko_u::PackageInputKind kInputRoutes[] = {
    hiko_u::PackageInputKind::StructureBackboneAtoms,
    hiko_u::PackageInputKind::CoordsBackbone,
    hiko_u::PackageInputKind::ResidueEmbeddings,
};

constexpr hiko_u::PackagePreprocessingCapability kPreprocessing[] = {
    hiko_u::PackagePreprocessingCapability::AtomInference,
    hiko_u::PackagePreprocessingCapability::VirtualCb,
    hiko_u::PackagePreprocessingCapability::CaKnn,
    hiko_u::PackagePreprocessingCapability::AtomPairDistances,
    hiko_u::PackagePreprocessingCapability::RbfExpand,
    hiko_u::PackagePreprocessingCapability::PositionalEncoding,
};

constexpr hiko_u::PackageOutputKind kOutputs[] = {
    hiko_u::PackageOutputKind::ResidueEmbeddings,
};

constexpr hiko_u::DataType kDtypes[] = {
    hiko_u::DataType::Float32,
};

constexpr hiko_u::PackageTensorLayout kLayouts[] = {
    hiko_u::PackageTensorLayout::RowMajor,
};

constexpr hiko_u::PackageBackendRequirement kBackends[] = {
    hiko_u::PackageBackendRequirement::CpuScalar,
};

constexpr hiko_u::ScoreInputKind kScoreInputs[] = {
    hiko_u::ScoreInputKind::ResidueEmbeddings,
};

constexpr hiko_u::TracebackPolicy kTracebacks[] = {
    hiko_u::TracebackPolicy::RequiredForPublicPairwise,
};

hiko_u::PackageDescriptor package_descriptor_fixture() {
  hiko_u::PackageDescriptor descriptor{};
  descriptor.identity = {"0.1.0",
                         hiko_w::kDefaultMpnnD64ModelName,
                         hiko_w::kDefaultMpnn64ModelFamily,
                         "0.1.0",
                         hiko_u::PackageKind::RegisteredArchitecture,
                         {nullptr, 0}};
  descriptor.execution = {hiko_u::PackageExecutionMode::RegisteredArchitecture,
                          "hikoboshi_mpnn_v1",
                          {kBackends, 1}};
  descriptor.capabilities = {0,
                             {kInputRoutes, 3},
                             {kPreprocessing, 6},
                             {kOutputs, 1},
                             {kDtypes, 1},
                             {kLayouts, 1},
                             {kBackends, 1}};
  descriptor.inputs = {{kInputRoutes, 3}};
  descriptor.outputs = {{kOutputs, 1}};
  descriptor.scoring = {hiko_u::ScoreMethod::RawDotV1,
                        {kScoreInputs, 1},
                        hiko_u::ScoreOutputKind::ScoreMatrix,
                        hiko_u::kRawDotV1ScoreSemantics};
    descriptor.gaps = {"Hikoboshi 0.1.0 hard-SW",
                       hiko_u::GapModel::Affine,
                       hiko::kDefaultGapOpen,
                       hiko::kDefaultGapExtension,
                       hiko_u::GapConvention::GapOpenPlusKMinusOneGapExtension,
                       hiko_u::ScoreMethod::RawDotV1};
    descriptor.soft_gaps = {"Hikoboshi 0.1.0 soft-SW",
                            hiko_u::GapModel::Affine,
                            -3.21337F,
                            -0.111704F,
                            hiko_u::GapConvention::
                                GapOpenPlusKMinusOneGapExtension,
                            hiko_u::ScoreMethod::RawDotV1};
  descriptor.alignment = {hiko_u::AlignmentAlgorithmId::HardLocalAffineSwV1,
                          {kTracebacks, 1}};
  descriptor.compatibility_views = {{nullptr, nullptr}};
  return descriptor;
}

hiko_u::EmbeddingView embedding_view(const std::vector<float>& values,
                                  const std::vector<char>& codes,
                                  std::size_t residue_count,
                                  std::size_t dimension) {
  return {residue_count,
          dimension,
          {values.data(), values.size()},
          {codes.data(), codes.size()},
          {nullptr, 0}};
}

hiko_u::StructureView structure_view(const std::vector<float>& coordinates,
                                  const std::vector<hiko_u::AtomSource>& atom_sources,
                                  const std::vector<char>& residue_codes) {
  return {residue_codes.size(),
          {coordinates.data(), coordinates.size()},
          {atom_sources.data(), atom_sources.size()},
          {residue_codes.data(), residue_codes.size()},
          {nullptr, 0},
          "synthetic",
          {},
          {}};
}

std::vector<float> synthetic_coordinates() {
  return {
      0.0F, 0.0F, 0.0F,
      1.0F, 0.0F, 0.0F,
      2.0F, 0.0F, 0.0F,
      2.0F, 1.0F, 0.0F,
      1.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.5F,
      1.0F, 0.0F, 1.5F,
      2.0F, 0.0F, 1.5F,
      2.0F, 1.0F, 1.5F,
      1.0F, 1.0F, 1.5F,
      0.0F, 0.0F, 3.0F,
      1.0F, 0.0F, 3.0F,
      2.0F, 0.0F, 3.0F,
      2.0F, 1.0F, 3.0F,
      1.0F, 1.0F, 3.0F,
  };
}

class RecordingAllVsAllSink final : public hiko::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko::PairwiseResultRecord& record) override {
    records.push_back(record);
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::vector<hiko::PairwiseResultRecord> records;
};

class RecordingAlgorithmAllVsAllSink final : public hiko_g::PairwiseResultSink {
 public:
  hiko_u::Status receive(const hiko_g::PairwiseResultRecord& record) override {
    records.push_back(record);
    return {hiko_u::StatusCode::Ok, ""};
  }

  std::vector<hiko_g::PairwiseResultRecord> records;
};

hikoboshi::modules::Mpnn64Descriptor mpnn64_descriptor_fixture() {
  return {64, 64, 16, 3, 30.0F};
}

template <typename AlgorithmRecords>
void require_api_matches_algorithms(const hiko::AllVsAllResult& api_result,
                                    const AlgorithmRecords& algorithm_records) {
  if (api_result.records.size() != algorithm_records.size()) {
    fail("API all-vs-all record count must match algorithms output");
  }
  for (std::size_t index = 0; index < api_result.records.size(); ++index) {
    const hiko::PairwiseResultRecord& api_record = api_result.records[index];
    const hiko_g::PairwiseResultRecord& algorithm_record =
        algorithm_records[index];
    if (api_record.query_index != algorithm_record.query_index ||
        api_record.target_index != algorithm_record.target_index ||
        !nearly_equal(api_record.result.metrics.raw_sw_score,
                      algorithm_record.result.metrics.raw_sw_score) ||
        api_record.result.path.aligned_pairs !=
            algorithm_record.result.path.aligned_pairs ||
        api_record.result.path.steps.size() !=
            algorithm_record.result.path.steps.size()) {
      fail("API all-vs-all records must match algorithms output");
    }
  }
}

void test_embedding_pairwise_runs_through_engine() {
  const std::vector<float> query_values = {
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F,
  };
  const std::vector<float> target_values = query_values;
  const std::vector<char> query_codes = {'A', 'C', 'D'};
  const std::vector<char> target_codes = query_codes;

  const hiko::Engine engine;
  hiko::PairwiseEmbeddingRequest hard_request{
      embedding_view(query_values, query_codes, 3, 3),
      embedding_view(target_values, target_codes, 3, 3)};
  // Pin to hard mode: this test asserts the legacy hard-SW raw_sw_score
  // (3.0 for an identity-matrix scoring matrix). The Hikoboshi 0.1.0 default
  // remains hard Smith-Waterman, which is what this test validates.
  hard_request.mode = hiko::AlignmentMode::Hard;
  const auto result = engine.pairwise(hard_request);

  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("embedding pairwise request must return Ok");
  }
  if (!nearly_equal(result.value.metrics.raw_sw_score, 3.0)) {
    fail("embedding pairwise raw SW score mismatch");
  }
  if (result.value.path.aligned_pairs != 3 ||
      result.value.path.steps.size() != 3) {
    fail("embedding pairwise path must contain three aligned pairs");
  }
  if (!result.value.metrics.coverage_query.valid ||
      !nearly_equal(result.value.metrics.coverage_query.value, 1.0) ||
      !result.value.metrics.identity.valid ||
      !nearly_equal(result.value.metrics.identity.value, 1.0)) {
    fail("embedding pairwise metrics must carry coverage and identity");
  }
  if (result.value.metrics.rmsd.valid ||
      result.value.metrics.rmsd.reason !=
          hiko_u::MetricInvalidReason::MissingStructureMetadata) {
    fail("embedding-only structural metrics must be explicitly invalid");
  }
  if (!result.value.warnings.empty()) {
    fail("default embedding pairwise must not emit warnings");
  }
}

void test_embedding_all_vs_all_runs_through_engine() {
  const std::vector<float> first = {1.0F, 0.0F};
  const std::vector<float> second = {0.0F, 1.0F};
  const std::vector<float> third = {1.0F, 1.0F};
  const std::vector<char> code = {'A'};
  const hiko_u::EmbeddingView embeddings[] = {
      embedding_view(first, code, 1, 2),
      embedding_view(second, code, 1, 2),
      embedding_view(third, code, 1, 2),
  };

  hiko::AllVsAllEmbeddingRequest request{};
  request.embeddings = {embeddings, 3};

  const hiko::Engine engine;
  const auto collected = engine.collect_all_vs_all(request);
  if (collected.status.code != hiko_u::StatusCode::Ok) {
    fail("embedding all-vs-all collect must return Ok through Engine");
  }
  if (collected.value.records.size() != 3 ||
      collected.value.records[0].query_index != 0 ||
      collected.value.records[0].target_index != 1 ||
      collected.value.records[1].query_index != 0 ||
      collected.value.records[1].target_index != 2 ||
      collected.value.records[2].query_index != 1 ||
      collected.value.records[2].target_index != 2) {
    fail("embedding all-vs-all collect must preserve chartered pair order");
  }

  // Pin to hard mode: all-vs-all does not yet expose soft mode (a1d), so
  // the direct pairwise comparison must use hard mode to share scoring
  // semantics with `collected`.
  hiko::PairwiseEmbeddingRequest direct_request{embeddings[0], embeddings[2]};
  direct_request.mode = hiko::AlignmentMode::Hard;
  const auto direct_pairwise = engine.pairwise(direct_request);
  if (direct_pairwise.status.code != hiko_u::StatusCode::Ok ||
      !nearly_equal(collected.value.records[1].result.metrics.raw_sw_score,
                    direct_pairwise.value.metrics.raw_sw_score)) {
    fail("embedding all-vs-all must share pairwise scoring semantics");
  }
  for (const hiko::PairwiseResultRecord& record : collected.value.records) {
    if (!record.result.warnings.empty()) {
      fail("default embedding all-vs-all must not emit API warnings");
    }
  }

  hiko_g::AllVsAllEmbeddingRequest algorithm_request{};
  algorithm_request.embeddings = {embeddings, 3};
  RecordingAlgorithmAllVsAllSink algorithm_sink;
  const hiko_u::Status algorithm_status =
      hiko_g::run_all_vs_all_embeddings(algorithm_request, algorithm_sink);
  if (algorithm_status.code != hiko_u::StatusCode::Ok) {
    fail("algorithms embedding all-vs-all fixture must return Ok");
  }
  require_api_matches_algorithms(collected.value, algorithm_sink.records);

  RecordingAllVsAllSink sink;
  const hiko_u::Status stream_status = engine.all_vs_all(request, sink);
  if (stream_status.code != hiko_u::StatusCode::Ok || sink.records.size() != 3) {
    fail("embedding all-vs-all streaming must return records through Engine");
  }
}

void test_embedding_gap_override_warning() {
  const std::vector<float> values = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C'};
  hiko::PairwiseEmbeddingRequest request{
      embedding_view(values, codes, 2, 2),
      embedding_view(values, codes, 2, 2)};
  request.alignment.gap_open = -2.0F;

  const hiko::Engine engine;
  const auto result = engine.pairwise(request);
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("embedding pairwise gap override must still return Ok");
  }
  if (result.value.warnings.size() != 1 ||
      result.value.warnings[0].kind !=
          hiko_u::PackageWarningKind::GapDefaultsOverridden ||
      result.value.warnings[0].code != "gap_defaults_overridden") {
    fail("gap override warning must be structured on the API result");
  }

  const hiko_u::EmbeddingView embeddings[] = {request.query, request.target};
  hiko::AllVsAllEmbeddingRequest all_vs_all_request{};
  all_vs_all_request.embeddings = {embeddings, 2};
  all_vs_all_request.options.alignment = request.alignment;
  const auto all_vs_all = engine.collect_all_vs_all(all_vs_all_request);
  if (all_vs_all.status.code != hiko_u::StatusCode::Ok ||
      all_vs_all.value.records.size() != 1 ||
      all_vs_all.value.records[0].result.warnings.size() != 1 ||
      all_vs_all.value.records[0].result.warnings[0].kind !=
          hiko_u::PackageWarningKind::GapDefaultsOverridden ||
      all_vs_all.value.records[0].result.warnings[0].code !=
          "gap_defaults_overridden") {
    fail("gap override warning must be structured on API all-vs-all records");
  }
}

void test_embedding_pairwise_ignores_invalid_package_config() {
  hiko_u::PackageDescriptor descriptor = package_descriptor_fixture();
  descriptor.execution.mode = hiko_u::PackageExecutionMode::GraphIr;
  hiko::EngineConfig config{};
  config.package = {nullptr, &descriptor};
  const hiko::Engine engine{config};

  const std::vector<float> values = {1.0F, 0.0F};
  const std::vector<char> codes = {'A'};
  const hiko_u::EmbeddingView embedding = embedding_view(values, codes, 1, 2);
  const auto result = engine.pairwise(
      hiko::PairwiseEmbeddingRequest{embedding, embedding});
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("embedding-only pairwise must not require model package validation");
  }
  const hiko_u::EmbeddingView embeddings[] = {embedding, embedding};
  const auto all_vs_all = engine.collect_all_vs_all(
      hiko::AllVsAllEmbeddingRequest{{embeddings, 2}});
  if (all_vs_all.status.code != hiko_u::StatusCode::Ok) {
    fail("embedding-only all-vs-all must not require model package validation");
  }
}

void test_planner_policy_axis_is_scalar_only() {
  const std::vector<float> values = {
      1.0F, 0.0F,
      0.0F, 1.0F,
  };
  const std::vector<char> codes = {'A', 'C'};
  const hiko::PairwiseEmbeddingRequest request{
      embedding_view(values, codes, 2, 2),
      embedding_view(values, codes, 2, 2)};

  hiko::EngineConfig scalar_config{};
  scalar_config.planner_policy = &hiko_u::kScalarDefaultPlannerPolicy;
  const hiko::Engine scalar_engine{scalar_config};
  const auto scalar_result = scalar_engine.pairwise(request);
  if (scalar_result.status.code != hiko_u::StatusCode::Ok) {
    fail("scalar default planner policy must be accepted");
  }

  hiko_u::Backend future_fallback_order[] = {hiko_u::Backend::Cuda};
  hiko_u::PlannerPolicy future_policy = hiko_u::kScalarDefaultPlannerPolicy;
  future_policy.fallback_order = {future_fallback_order, 1};
  future_policy.default_backend = hiko_u::Backend::Cuda;

  hiko::EngineConfig future_config{};
  future_config.planner_policy = &future_policy;
  const hiko::Engine future_engine{future_config};
  const auto future_result = future_engine.pairwise(request);
  if (future_result.status.code != hiko_u::StatusCode::InvalidArgument ||
      !contains(detail(future_result.status), "planner policy")) {
    fail("non-scalar planner policy must be rejected in 0.1.0");
  }

  const hiko_u::EmbeddingView embeddings[] = {request.query, request.target};
  const auto future_all_vs_all = future_engine.collect_all_vs_all(
      hiko::AllVsAllEmbeddingRequest{{embeddings, 2}});
  if (future_all_vs_all.status.code != hiko_u::StatusCode::InvalidArgument ||
      !contains(detail(future_all_vs_all.status), "planner policy")) {
    fail("all-vs-all must reject non-scalar planner policies");
  }

  hiko::EngineConfig future_backend_config{};
  future_backend_config.execution.backend = hiko_u::Backend::Cuda;
  const hiko::Engine future_backend_engine{future_backend_config};
  const auto future_backend_all_vs_all =
      future_backend_engine.collect_all_vs_all(
          hiko::AllVsAllEmbeddingRequest{{embeddings, 2}});
  if (future_backend_all_vs_all.status.code !=
          hiko_u::StatusCode::InvalidArgument ||
      !contains(detail(future_backend_all_vs_all.status), "backend")) {
    fail("all-vs-all must reject unimplemented backend selections");
  }
}

void test_structure_request_requires_weight_handle() {
  constexpr std::size_t kResidues = 1;
  std::vector<float> coordinates(kResidues * hiko_u::kCanonicalAtomCount *
                                 hiko_u::kCoordinateAxisCount);
  std::vector<hiko_u::AtomSource> atom_sources(kResidues * hiko_u::kCanonicalAtomCount,
                                            hiko_u::AtomSource::Observed);
  std::vector<char> residue_codes = {'A'};
  const hiko_u::StructureView structure =
      structure_view(coordinates, atom_sources, residue_codes);

  const hiko::Engine engine;
  const auto result =
      engine.pairwise(hiko::PairwiseStructureRequest{structure, structure});
  if (result.status.code != hiko_u::StatusCode::FailedPrecondition) {
    fail("structure pairwise without weights must fail deterministically");
  }
  const hiko_u::StructureView structures[] = {structure, structure};
  const auto all_vs_all =
      engine.collect_all_vs_all(hiko::AllVsAllStructureRequest{{structures, 2}});
  if (all_vs_all.status.code != hiko_u::StatusCode::FailedPrecondition) {
    fail("structure all-vs-all without weights must fail deterministically");
  }
}

void test_structure_config_precedence_axes() {
  const std::vector<float> coordinates = synthetic_coordinates();
  std::vector<hiko_u::AtomSource> atom_sources(3 * hiko_u::kCanonicalAtomCount,
                                            hiko_u::AtomSource::Observed);
  const std::vector<char> residue_codes = {'A', 'C', 'D'};
  const hiko_u::StructureView structure =
      structure_view(coordinates, atom_sources, residue_codes);

  const hiko_u::Result<hiko_w::PackageHandle> package_result =
      hiko_w::default_mpnn_d64_package();
  if (package_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.descriptor == nullptr) {
    fail("default package must be available for EngineConfig precedence test");
  }

  hiko::EngineConfig package_config{};
  package_config.package = package_result.value;
  package_config.execution.backend = hiko_u::Backend::Scalar;
  const hiko::Engine package_engine{package_config};
  const auto package_pairwise =
      package_engine.pairwise(hiko::PairwiseStructureRequest{structure, structure});
  if (package_pairwise.status.code != hiko_u::StatusCode::Ok) {
    fail("EngineConfig package handle must supply compatibility weights");
  }

  hiko::EngineConfig weights_config{};
  weights_config.weights =
      package_result.value.descriptor->compatibility_views.weights;
  weights_config.execution.backend = hiko_u::Backend::Scalar;
  const hiko::Engine weights_engine{weights_config};
  const auto weights_pairwise =
      weights_engine.pairwise(hiko::PairwiseStructureRequest{structure, structure});
  if (weights_pairwise.status.code != hiko_u::StatusCode::Ok) {
    fail("EngineConfig compatibility weights handle must remain accepted");
  }

  if (!nearly_equal(package_pairwise.value.metrics.raw_sw_score,
                    weights_pairwise.value.metrics.raw_sw_score)) {
    fail("package and weights compatibility construction must agree");
  }
}

void expect_structure_package_rejection(hiko_u::PackageDescriptor& descriptor,
                                        std::string_view expected_detail) {
  constexpr std::size_t kResidues = 1;
  std::vector<float> coordinates(kResidues * hiko_u::kCanonicalAtomCount *
                                 hiko_u::kCoordinateAxisCount);
  std::vector<hiko_u::AtomSource> atom_sources(kResidues * hiko_u::kCanonicalAtomCount,
                                            hiko_u::AtomSource::Observed);
  std::vector<char> residue_codes = {'A'};
  const hiko_u::StructureView structure =
      structure_view(coordinates, atom_sources, residue_codes);

  hiko::EngineConfig config{};
  config.package = {nullptr, &descriptor};
  const hiko::Engine engine{config};
  const auto result =
      engine.pairwise(hiko::PairwiseStructureRequest{structure, structure});
  if (result.status.code != hiko_u::StatusCode::InvalidArgument ||
      !contains(detail(result.status), expected_detail)) {
    fail("unsupported package route must return a structured diagnostic");
  }
  const hiko_u::StructureView structures[] = {structure, structure};
  const auto all_vs_all =
      engine.collect_all_vs_all(hiko::AllVsAllStructureRequest{{structures, 2}});
  if (all_vs_all.status.code != hiko_u::StatusCode::InvalidArgument ||
      !contains(detail(all_vs_all.status), expected_detail)) {
    fail("all-vs-all must use the same package rejection as pairwise");
  }
}

void test_structure_package_validation_rejects_reserved_routes() {
  {
    hiko_u::PackageDescriptor descriptor = package_descriptor_fixture();
    descriptor.execution.mode = hiko_u::PackageExecutionMode::GraphIr;
    expect_structure_package_rejection(descriptor, "graph_ir");
  }
  {
    static constexpr hiko_u::PackageInputKind kAllAtomOnly[] = {
        hiko_u::PackageInputKind::StructureAllAtom,
    };
    hiko_u::PackageDescriptor descriptor = package_descriptor_fixture();
    descriptor.inputs.routes = {kAllAtomOnly, 1};
    descriptor.capabilities.input_routes = {kAllAtomOnly, 1};
    expect_structure_package_rejection(descriptor, "structure_all_atom");
  }
  {
    hiko_u::PackageDescriptor descriptor = package_descriptor_fixture();
    descriptor.scoring.method = hiko_u::ScoreMethod::CosineV1;
    descriptor.scoring.semantics.method = hiko_u::ScoreMethod::CosineV1;
    expect_structure_package_rejection(descriptor, "cosine");
  }
  {
    hiko_u::PackageDescriptor descriptor = package_descriptor_fixture();
    descriptor.alignment.algorithm = hiko_u::AlignmentAlgorithmId::SoftSwV1;
    expect_structure_package_rejection(descriptor, "hard_local_affine_sw_v1");
  }
}

void test_encode_validates_package_before_mpnn() {
  constexpr std::size_t kResidues = 1;
  std::vector<float> coordinates(kResidues * hiko_u::kCanonicalAtomCount *
                                 hiko_u::kCoordinateAxisCount);
  std::vector<hiko_u::AtomSource> atom_sources(kResidues * hiko_u::kCanonicalAtomCount,
                                            hiko_u::AtomSource::Observed);
  std::vector<char> residue_codes = {'A'};
  const hiko_u::StructureView structure =
      structure_view(coordinates, atom_sources, residue_codes);

  hiko_u::PackageDescriptor descriptor = package_descriptor_fixture();
  descriptor.execution.mode = hiko_u::PackageExecutionMode::GraphIr;
  hiko::EngineConfig config{};
  config.package = {nullptr, &descriptor};
  const hiko::Engine engine{config};
  const auto result = engine.encode(hiko::EncodeStructureRequest{structure});
  if (result.status.code != hiko_u::StatusCode::InvalidArgument ||
      !contains(detail(result.status), "graph_ir")) {
    fail("encode must validate unsupported package routes before MPNN execution");
  }
}

void test_structure_and_coords_all_vs_all_share_pairwise_path() {
  const std::vector<float> coordinates = synthetic_coordinates();
  std::vector<hiko_u::AtomSource> atom_sources(3 * hiko_u::kCanonicalAtomCount,
                                            hiko_u::AtomSource::Observed);
  const std::vector<char> residue_codes = {'A', 'C', 'D'};
  const hiko_u::StructureView structure =
      structure_view(coordinates, atom_sources, residue_codes);
  const hiko::CoordsInputView coords{
      structure.residue_count,
      structure.coordinates,
      structure.atom_sources,
      structure.residue_codes,
      structure.residues,
  };

  const hiko_u::Result<hiko_w::PackageHandle> package_result =
      hiko_w::default_mpnn_d64_package();
  if (package_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.descriptor == nullptr) {
    fail("default hikoboshi-mpnn-d64 package must validate for all-vs-all");
  }
  hiko::EngineConfig config{};
  config.package = package_result.value;
  config.weights =
      package_result.value.descriptor->compatibility_views.weights;
  config.execution.backend = hiko_u::Backend::Scalar;
  const hiko::Engine engine{config};

  // Pin pairwise to hard mode so it shares scoring semantics with the
  // hard-default all-vs-all path (soft all-vs-all is a1d / out of scope here).
  hiko::PairwiseStructureRequest pairwise_structure_request{structure, structure};
  pairwise_structure_request.mode = hiko::AlignmentMode::Hard;
  const auto pairwise_structure = engine.pairwise(pairwise_structure_request);
  if (pairwise_structure.status.code != hiko_u::StatusCode::Ok) {
    fail("structure pairwise with default package must return Ok");
  }

  const hiko_u::StructureView structures[] = {structure, structure};
  const auto all_structures =
      engine.collect_all_vs_all(hiko::AllVsAllStructureRequest{{structures, 2}});
  if (all_structures.status.code != hiko_u::StatusCode::Ok ||
      all_structures.value.records.size() != 1 ||
      all_structures.value.records[0].query_index != 0 ||
      all_structures.value.records[0].target_index != 1) {
    fail("structure all-vs-all with default package must emit one pair");
  }
  if (!nearly_equal(
          all_structures.value.records[0].result.metrics.raw_sw_score,
          pairwise_structure.value.metrics.raw_sw_score) ||
      all_structures.value.records[0].result.path.aligned_pairs !=
          pairwise_structure.value.path.aligned_pairs) {
    fail("structure all-vs-all must share pairwise score and alignment path");
  }

  hiko_g::AllVsAllStructureRequest algorithm_structure_request{};
  algorithm_structure_request.structures = {structures, 2};
  algorithm_structure_request.descriptor = mpnn64_descriptor_fixture();
  using PreparedWeightsPtr =
      decltype(hiko_g::AllVsAllStructureRequest{}.weights);
  algorithm_structure_request.weights = static_cast<PreparedWeightsPtr>(
      package_result.value.descriptor->compatibility_views.weights.opaque);
  RecordingAlgorithmAllVsAllSink algorithm_structure_sink;
  const hiko_u::Status algorithm_structure_status =
      hiko_g::run_all_vs_all_structures(algorithm_structure_request,
                                     algorithm_structure_sink);
  if (algorithm_structure_status.code != hiko_u::StatusCode::Ok) {
    fail("algorithms structure all-vs-all fixture must return Ok");
  }
  require_api_matches_algorithms(all_structures.value,
                                 algorithm_structure_sink.records);

  const hiko::CoordsInputView coord_inputs[] = {coords, coords};
  // Same hard-mode pin for the coords-path comparison against hard all-vs-all.
  hiko::PairwiseCoordsRequest pairwise_coords_request{coords, coords};
  pairwise_coords_request.mode = hiko::AlignmentMode::Hard;
  const auto pairwise_coords = engine.pairwise(pairwise_coords_request);
  const auto all_coords =
      engine.collect_all_vs_all(hiko::AllVsAllCoordsRequest{{coord_inputs, 2}});
  if (pairwise_coords.status.code != hiko_u::StatusCode::Ok ||
      all_coords.status.code != hiko_u::StatusCode::Ok ||
      all_coords.value.records.size() != 1 ||
      !nearly_equal(all_coords.value.records[0].result.metrics.raw_sw_score,
                    pairwise_coords.value.metrics.raw_sw_score) ||
      all_coords.value.records[0].result.path.aligned_pairs !=
          pairwise_coords.value.path.aligned_pairs) {
    fail("coords all-vs-all must share pairwise score and alignment path");
  }
  require_api_matches_algorithms(all_coords.value,
                                 algorithm_structure_sink.records);
}

}  // namespace

int main() {
  test_embedding_pairwise_runs_through_engine();
  test_embedding_all_vs_all_runs_through_engine();
  test_embedding_gap_override_warning();
  test_embedding_pairwise_ignores_invalid_package_config();
  test_planner_policy_axis_is_scalar_only();
  test_structure_request_requires_weight_handle();
  test_structure_config_precedence_axes();
  test_structure_package_validation_rejects_reserved_routes();
  test_encode_validates_package_before_mpnn();
  test_structure_and_coords_all_vs_all_share_pairwise_path();
  return 0;
}
