#ifndef HIKOBOSHI_UNIVERSAL_PACKAGE_HPP
#define HIKOBOSHI_UNIVERSAL_PACKAGE_HPP

/// @file
/// Public model/scoring package descriptors.
///
/// Hikoboshi 0.1.0 ships a compiled hikoboshi-mpnn-d64 package that produces
/// residue embeddings, scores residue pairs with raw dot products, and aligns
/// those scores with hard local affine Smith-Waterman. The descriptor types
/// also carry reserved vocabulary for diagnostics; reserved values are
/// rejected unless a build explicitly implements them.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>
#include <hikoboshi/universal/weights.hpp>

namespace hikoboshi::universal {

/// Package storage and execution family.
enum class PackageKind : std::uint8_t {
  Unknown = 0,
  RegisteredArchitecture = 1,
  GraphIr = 2,
  SubstitutionMatrix = 3,
};

/// How a package's executable state is represented.
enum class PackageExecutionMode : std::uint8_t {
  Unknown = 0,
  RegisteredArchitecture = 1,
  GraphIr = 2,
};

/// Backend family required by a package.
enum class PackageBackendRequirement : std::uint8_t {
  CpuScalar = 0,
  GpuCuda = 1,
  GpuMetal = 2,
  GpuHip = 3,
};

/// Input route accepted or described by a package.
enum class PackageInputKind : std::uint8_t {
  StructureBackboneAtoms = 0,
  CoordsBackbone = 1,
  ResidueEmbeddings = 2,
  StructureAllAtom = 3,
  SequenceTokens = 4,
  DirectScoreMatrix = 5,
};

/// Preprocessing capability needed before package execution.
enum class PackagePreprocessingCapability : std::uint8_t {
  AtomInference = 0,
  VirtualCb = 1,
  CaKnn = 2,
  AtomPairDistances = 3,
  RbfExpand = 4,
  PositionalEncoding = 5,
  Tokenization = 6,
};

/// Output kind a package can produce.
enum class PackageOutputKind : std::uint8_t {
  ResidueEmbeddings = 0,
  SubstitutionScores = 1,
  DirectPairScores = 2,
};

/// Tensor memory layout tag used by package descriptors.
enum class PackageTensorLayout : std::uint8_t {
  RowMajor = 0,
};

/// Score construction method declared by a package.
///
/// hikoboshi-mpnn-d64 uses `RawDotV1`: `S_AB[p, q]` is the dot product of query
/// and target residue embeddings without cosine normalization.
enum class ScoreMethod : std::uint8_t {
  RawDotV1 = 0,
  CosineV1 = 1,
  SubstitutionLookupV1 = 2,
  DirectScoreMatrixV1 = 3,
  LearnedPairScorerV1 = 4,
};

/// Source data consumed by a scoring method.
enum class ScoreInputKind : std::uint8_t {
  ResidueEmbeddings = 0,
  SequenceTokens = 1,
  DirectPairInput = 2,
  DirectScoreMatrix = 3,
};

/// Scoring output family consumed by alignment.
enum class ScoreOutputKind : std::uint8_t {
  ScoreMatrix = 0,
};

/// Score-matrix memory layout.
enum class ScoreMatrixLayout : std::uint8_t {
  RowMajorQueryByTarget = 0,
};

/// Borrowed score matrix used by the alignment boundary.
///
/// Values are indexed as `values[row * row_stride + column]`, where rows are
/// query residues and columns are target residues. Hikoboshi 0.1.0 expects
/// float32 raw-dot scores with higher values indicating better residue pairs.
struct ScoreMatrixView {
  const float* values;
  std::size_t query_length;
  std::size_t target_length;
  std::size_t row_stride;
};

/// Normalization applied before scores enter alignment.
enum class ScoreNormalization : std::uint8_t {
  None = 0,
  L2 = 1,
  CalibratedLogOdds = 2,
  PackageSpecific = 3,
};

/// Numeric family for interpreting score magnitudes.
enum class ScoreScaleFamily : std::uint8_t {
  RawDot = 0,
  CosineUnitless = 1,
  LogOdds = 2,
  LearnedLogit = 3,
  Unknown = 4,
};

/// Alignment gap model family.
enum class GapModel : std::uint8_t {
  Affine = 0,
};

/// Convention used to score a gap of length `k`.
enum class GapConvention : std::uint8_t {
  GapOpenPlusKMinusOneGapExtension = 0,
};

/// Numeric affine gap parameters.
///
/// Under the Hikoboshi convention, a gap of length `k` costs
/// `gap_open + (k - 1) * gap_extension`.
struct AffineGapModel {
  GapModel model;
  float gap_open;
  float gap_extension;
  GapConvention convention;
};

/// Alignment algorithm declared by a package.
///
/// `HardLocalAffineSwV1` is the only implemented Hikoboshi 0.1.0 public
/// alignment algorithm. Other values are reserved descriptor vocabulary.
enum class AlignmentAlgorithmId : std::uint8_t {
  HardLocalAffineSwV1 = 0,
  GlobalAffineSwV1 = 1,
  SemiglobalAffineSwV1 = 2,
  SoftSwV1 = 3,
};

/// Traceback requirement for a public workflow.
enum class TracebackPolicy : std::uint8_t {
  Unspecified = 0,
  RequiredForPublicPairwise = 1,
  RequiredForPublicAllVsAll = 2,
};

/// Validation stage used in package diagnostics and reports.
enum class PackageValidationStage : std::uint8_t {
  SchemaVersion = 0,
  StorageChecksum = 1,
  ArchitectureRegistration = 2,
  TensorTableRolesShapesDtypes = 3,
  InputRoute = 4,
  PreprocessingCapabilities = 5,
  ScoringMethod = 6,
  ScoreMatrixSemantics = 7,
  GapModelDefaults = 8,
  AlignmentAlgorithm = 9,
  WorkflowCompatibility = 10,
  PreparedStateBuild = 11,
};

/// Severity for package validation diagnostics.
enum class PackageDiagnosticSeverity : std::uint8_t {
  Info = 0,
  Warning = 1,
  Error = 2,
};

/// Structured warning emitted while resolving or executing a package.
enum class PackageWarningKind : std::uint8_t {
  Unspecified = 0,
  GapDefaultsOverridden = 1,
  UnsupportedReservedCapability = 2,
  IgnoredHistoricalTensor = 3,
};

/// Bit flags summarizing package routes, preprocessing, outputs, and backend.
enum class PackageCapabilityFlag : std::uint64_t {
  None = 0,
  StructureBackboneAtoms = 1ull << 0,
  CoordsBackbone = 1ull << 1,
  ResidueEmbeddings = 1ull << 2,
  AtomInference = 1ull << 16,
  VirtualCb = 1ull << 17,
  CaKnn = 1ull << 18,
  AtomPairDistances = 1ull << 19,
  RbfExpand = 1ull << 20,
  PositionalEncoding = 1ull << 21,
  OutputResidueEmbeddings = 1ull << 32,
  BackendCpuScalar = 1ull << 48,
};

using PackageCapabilityFlags = std::uint64_t;

/// Stable package identity and user-facing aliases.
///
/// `package_id` is canonical, for example `hikoboshi-mpnn-d64`. Aliases are
/// convenience names such as `mpnn64`; they are not version strings.
struct PackageIdentity {
  std::string_view package_schema_version;
  std::string_view package_id;
  std::string_view package_family;
  std::string_view package_version;
  PackageKind package_kind;
  Span<const std::string_view> aliases;
};

/// Executable package mode and backend requirements.
struct PackageExecution {
  PackageExecutionMode mode;
  std::string_view architecture_id;
  Span<const PackageBackendRequirement> backend_requirements;
};

/// Package input routes declared by the descriptor.
struct PackageInputs {
  Span<const PackageInputKind> routes;
};

/// Package output kinds declared by the descriptor.
struct PackageOutputs {
  Span<const PackageOutputKind> kinds;
};

/// Flat capability summary plus detailed route lists.
struct PackageCapabilities {
  PackageCapabilityFlags flags;
  Span<const PackageInputKind> input_routes;
  Span<const PackagePreprocessingCapability> preprocessing;
  Span<const PackageOutputKind> output_kinds;
  Span<const DataType> dtypes;
  Span<const PackageTensorLayout> layouts;
  Span<const PackageBackendRequirement> backends;
};

/// Semantics of the score matrix handed to the alignment boundary.
struct ScoreSemantics {
  DataType dtype;
  ScoreMatrixLayout layout;
  bool higher_is_better;
  bool local_affine_additive;
  ScoreNormalization normalization;
  ScoreScaleFamily scale_family;
  ScoreMethod method = ScoreMethod::RawDotV1;
};

/// Canonical raw-dot score semantics for hikoboshi-mpnn-d64 residue embeddings.
inline constexpr ScoreSemantics kRawDotV1ScoreSemantics{
    DataType::Float32,
    ScoreMatrixLayout::RowMajorQueryByTarget,
    true,
    true,
    ScoreNormalization::None,
    ScoreScaleFamily::RawDot,
    ScoreMethod::RawDotV1,
};

/// Canonical hard-SW affine gap defaults for Hikoboshi 0.1.0.
inline constexpr AffineGapModel kHardSwDefaultAffineGapModel{
    GapModel::Affine,
    -1.40000F,
    -0.150000F,
    GapConvention::GapOpenPlusKMinusOneGapExtension,
};

/// Package scoring contract.
struct ScoringDescriptor {
  ScoreMethod method;
  Span<const ScoreInputKind> inputs;
  ScoreOutputKind output;
  ScoreSemantics semantics;
};

/// User-visible affine gap defaults calibrated for one score method.
struct AffineGapDefaults {
  std::string_view family;
  GapModel model;
  float gap_open;
  float gap_extension;
  GapConvention convention;
  ScoreMethod calibrated_for_score_method;
};

using GapDescriptor = AffineGapDefaults;

/// Alignment behavior promised by a package.
struct PackageAlignmentDescriptor {
  AlignmentAlgorithmId algorithm;
  Span<const TracebackPolicy> traceback_policies;
};

/// Compatibility views exposed beside the package descriptor.
struct PackageCompatibilityViews {
  /// Legacy tensor view for callers that do not need package semantics.
  WeightsHandle weights;
};

/// Complete public descriptor for a compiled or otherwise registered package.
struct PackageDescriptor {
  PackageIdentity identity;
  PackageExecution execution;
  PackageCapabilities capabilities;
  PackageInputs inputs;
  PackageOutputs outputs;
  ScoringDescriptor scoring;
  /// Release-default hard-SW gap family. This remains the Hikoboshi 0.1.0
  /// product-line default.
  AffineGapDefaults gaps;
  /// Benchmark/research soft-SW companion gap family. Soft-mode execution uses
  /// these values unless the caller explicitly overrides gap fields.
  AffineGapDefaults soft_gaps;
  PackageAlignmentDescriptor alignment;
  PackageCompatibilityViews compatibility_views;
};

/// Opaque package handle plus descriptor pointer.
///
/// The descriptor names the package and its semantics; `opaque` points to
/// provider-owned prepared state. Callers must not inspect `opaque`.
struct PackageHandle {
  const void* opaque;
  const PackageDescriptor* descriptor;
};

/// Non-fatal package warning returned through API results and validation.
struct PackageWarning {
  PackageWarningKind kind;
  PackageValidationStage stage;
  std::string_view code;
  std::string_view message;
};

/// Package validation diagnostic entry.
struct PackageValidationDiagnostic {
  PackageDiagnosticSeverity severity;
  PackageValidationStage stage;
  std::string_view code;
  std::string_view message;
};

/// Package validation outcome with caller-owned diagnostic storage.
///
/// Stage flags use `1ull << static_cast<unsigned>(PackageValidationStage)`.
/// `ok` indicates whether the accepted handle is usable for Hikoboshi 0.1.0
/// workflows.
struct PackageValidationReport {
  PackageHandle accepted_handle;
  Span<const PackageValidationDiagnostic> diagnostics;
  Span<const PackageWarning> warnings;
  std::uint64_t passed_stage_flags;
  std::uint64_t failed_stage_flags;
  bool ok;
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_PACKAGE_HPP
