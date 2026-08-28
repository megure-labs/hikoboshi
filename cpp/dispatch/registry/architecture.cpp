#include <hikoboshi/dispatch/registry/architecture.hpp>

#include <cstddef>
#include <string_view>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::dispatch::registry {
namespace {

namespace hiko_u = hikoboshi::universal;

constexpr std::string_view kHikoboshiMpnnV1ArchitectureId{"hikoboshi_mpnn_v1"};
constexpr std::string_view kHikoboshiMpnnV1ModuleOpKind{"mpnn_v1_encoder"};
constexpr std::string_view kHikoboshiMpnnV1ScoringOpKind{"dot_product_similarity_v1"};
constexpr std::string_view kHikoboshiMpnnV1PreparedStateKind{"prepared_mpnn64"};

constexpr hiko_u::PackageInputKind kMpnnV1InputRoutes[] = {
    hiko_u::PackageInputKind::StructureBackboneAtoms,
    hiko_u::PackageInputKind::CoordsBackbone,
    hiko_u::PackageInputKind::ResidueEmbeddings,
};

constexpr hiko_u::PackagePreprocessingCapability kMpnnV1Preprocessing[] = {
    hiko_u::PackagePreprocessingCapability::AtomInference,
    hiko_u::PackagePreprocessingCapability::VirtualCb,
    hiko_u::PackagePreprocessingCapability::CaKnn,
    hiko_u::PackagePreprocessingCapability::AtomPairDistances,
    hiko_u::PackagePreprocessingCapability::RbfExpand,
    hiko_u::PackagePreprocessingCapability::PositionalEncoding,
};

constexpr hiko_u::PackageOutputKind kMpnnV1OutputKinds[] = {
    hiko_u::PackageOutputKind::ResidueEmbeddings,
};

constexpr hiko_u::DataType kMpnnV1Dtypes[] = {
    hiko_u::DataType::Float32,
};

constexpr hiko_u::PackageTensorLayout kMpnnV1Layouts[] = {
    hiko_u::PackageTensorLayout::RowMajor,
};

constexpr hiko_u::PackageBackendRequirement kMpnnV1Backends[] = {
    hiko_u::PackageBackendRequirement::CpuScalar,
};

constexpr hiko_u::PackageCapabilityFlags kMpnnV1CapabilityFlags =
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::StructureBackboneAtoms) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::CoordsBackbone) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::ResidueEmbeddings) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::AtomInference) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::VirtualCb) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::CaKnn) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::AtomPairDistances) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::RbfExpand) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::PositionalEncoding) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::OutputResidueEmbeddings) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::BackendCpuScalar);

constexpr hiko_u::PackageCapabilities kMpnnV1Capabilities{
    kMpnnV1CapabilityFlags,
    {kMpnnV1InputRoutes, sizeof(kMpnnV1InputRoutes) / sizeof(kMpnnV1InputRoutes[0])},
    {kMpnnV1Preprocessing,
     sizeof(kMpnnV1Preprocessing) / sizeof(kMpnnV1Preprocessing[0])},
    {kMpnnV1OutputKinds,
     sizeof(kMpnnV1OutputKinds) / sizeof(kMpnnV1OutputKinds[0])},
    {kMpnnV1Dtypes, sizeof(kMpnnV1Dtypes) / sizeof(kMpnnV1Dtypes[0])},
    {kMpnnV1Layouts, sizeof(kMpnnV1Layouts) / sizeof(kMpnnV1Layouts[0])},
    {kMpnnV1Backends, sizeof(kMpnnV1Backends) / sizeof(kMpnnV1Backends[0])},
};

constexpr hiko_u::PackageInputs kMpnnV1Inputs{
    {kMpnnV1InputRoutes, sizeof(kMpnnV1InputRoutes) / sizeof(kMpnnV1InputRoutes[0])},
};

// Compound module-ops the MPNN-64 architecture composes. Each id must
// resolve in `module_op_registry()`. Hikoboshi 0.1.0 composes the MPNN-64
// encoder and the raw-dot similarity module; future architectures register
// additional module-op ids here.
constexpr std::string_view kMpnnV1RequiredModuleOpIds[] = {
    "hikoboshi.mpnn.v1.encoder",
    "hikoboshi.similarity.dot_product.v1",
};

// --------------------------------------------------------------------
// Hikoboshi-ESM2-8M architecture record (descriptor-only registration).
// The architecture is registered so the validation pipeline can resolve
// the `hikoboshi_esm2_v1` id at stage 2 and the closed-op-set capability
// surface for sequence-only architectures gains a live record. The
// `builder` pointer is left null in this packet; the
// `attention-compound-module` and `esm2-8m-forward` packets populate the
// real prepared-state builder. Compound module-op ids are reserved here
// so module-op registration packets can wire them up without a separate
// architecture edit.
// --------------------------------------------------------------------
constexpr std::string_view kHikoboshiEsm2V1ArchitectureId{"hikoboshi_esm2_v1"};
constexpr std::string_view kHikoboshiEsm2V1ModuleOpKind{"plm_v1_encoder"};
constexpr std::string_view kHikoboshiEsm2V1ScoringOpKind{
    "dot_product_similarity_v1"};
constexpr std::string_view kHikoboshiEsm2V1PreparedStateKind{
    "prepared_esm2_8m"};

constexpr hiko_u::PackageInputKind kEsm2V1InputRoutes[] = {
    hiko_u::PackageInputKind::SequenceTokens,
    hiko_u::PackageInputKind::ResidueEmbeddings,
};

constexpr hiko_u::PackagePreprocessingCapability kEsm2V1Preprocessing[] = {
    hiko_u::PackagePreprocessingCapability::Tokenization,
};

constexpr hiko_u::PackageOutputKind kEsm2V1OutputKinds[] = {
    hiko_u::PackageOutputKind::ResidueEmbeddings,
};

constexpr hiko_u::DataType kEsm2V1Dtypes[] = {
    hiko_u::DataType::Float32,
};

constexpr hiko_u::PackageTensorLayout kEsm2V1Layouts[] = {
    hiko_u::PackageTensorLayout::RowMajor,
};

constexpr hiko_u::PackageBackendRequirement kEsm2V1Backends[] = {
    hiko_u::PackageBackendRequirement::CpuScalar,
};

constexpr hiko_u::PackageCapabilityFlags kEsm2V1CapabilityFlags =
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::ResidueEmbeddings) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::OutputResidueEmbeddings) |
    static_cast<hiko_u::PackageCapabilityFlags>(
        hiko_u::PackageCapabilityFlag::BackendCpuScalar);

constexpr hiko_u::PackageCapabilities kEsm2V1Capabilities{
    kEsm2V1CapabilityFlags,
    {kEsm2V1InputRoutes,
     sizeof(kEsm2V1InputRoutes) / sizeof(kEsm2V1InputRoutes[0])},
    {kEsm2V1Preprocessing,
     sizeof(kEsm2V1Preprocessing) / sizeof(kEsm2V1Preprocessing[0])},
    {kEsm2V1OutputKinds,
     sizeof(kEsm2V1OutputKinds) / sizeof(kEsm2V1OutputKinds[0])},
    {kEsm2V1Dtypes, sizeof(kEsm2V1Dtypes) / sizeof(kEsm2V1Dtypes[0])},
    {kEsm2V1Layouts, sizeof(kEsm2V1Layouts) / sizeof(kEsm2V1Layouts[0])},
    {kEsm2V1Backends, sizeof(kEsm2V1Backends) / sizeof(kEsm2V1Backends[0])},
};

constexpr hiko_u::PackageInputs kEsm2V1Inputs{
    {kEsm2V1InputRoutes,
     sizeof(kEsm2V1InputRoutes) / sizeof(kEsm2V1InputRoutes[0])},
};

constexpr std::string_view kEsm2V1RequiredModuleOpIds[] = {
    "hikoboshi.esm2.v1.encoder",
    "hikoboshi.similarity.dot_product.v1",
};

// Builder thunk for the Hikoboshi-ESM2-8M architecture. The dispatch
// registry must observe a non-null builder so the
// `compound-module-architecture-agnostic-interface` constraint and the
// validation pipeline can both treat the architecture as live. The
// builder body itself is deliberately a no-op in this packet: ESM2-8M
// prepared-state construction (workspace allocation, RoPE table
// precompute, weight-view binding) happens inside the algorithms-layer
// `prepare_per_worker_*` helpers because the `PackageHandle` opaque is
// the embedded safetensors blob and the builder cannot allocate without
// reaching into modules-layer types from the dispatch TU. The thunk
// stays here as the registry-visible address; once the
// `esm2-8m-weights-package` packet populates real tensors the engine
// drives prepared-state allocation from its own arena.
void esm2_v1_builder(const universal::PackageHandle& package) noexcept {
  (void)package;
}

}  // namespace

universal::Span<const RegisteredArchitectureRecord>
architecture_registry() noexcept {
  static const RegisteredArchitectureRecord kRecords[] = {
      {
          ArchitectureKind::Mpnn64,
          kHikoboshiMpnnV1ArchitectureId,
          kHikoboshiMpnnV1ModuleOpKind,
          kHikoboshiMpnnV1ScoringOpKind,
          kHikoboshiMpnnV1PreparedStateKind,
          nullptr,
          &kMpnnV1Capabilities,
          &kMpnnV1Inputs,
          {kMpnnV1RequiredModuleOpIds,
           sizeof(kMpnnV1RequiredModuleOpIds) /
               sizeof(kMpnnV1RequiredModuleOpIds[0])},
      },
      {
          ArchitectureKind::Esm2_8m,
          kHikoboshiEsm2V1ArchitectureId,
          kHikoboshiEsm2V1ModuleOpKind,
          kHikoboshiEsm2V1ScoringOpKind,
          kHikoboshiEsm2V1PreparedStateKind,
          &esm2_v1_builder,
          &kEsm2V1Capabilities,
          &kEsm2V1Inputs,
          {kEsm2V1RequiredModuleOpIds,
           sizeof(kEsm2V1RequiredModuleOpIds) /
               sizeof(kEsm2V1RequiredModuleOpIds[0])},
      },
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

const RegisteredArchitectureRecord* find_architecture(
    const std::string_view architecture_id) noexcept {
  const universal::Span<const RegisteredArchitectureRecord> records =
      architecture_registry();
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].architecture_id == architecture_id) {
      return &records.data[index];
    }
  }
  return nullptr;
}

}  // namespace hikoboshi::dispatch::registry
