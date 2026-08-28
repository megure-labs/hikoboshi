#include <hikoboshi/dispatch/registry/capability.hpp>

#include <cstddef>

#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/tensor.hpp>

namespace hikoboshi::dispatch::registry {
namespace {

namespace hiko_u = hikoboshi::universal;

}  // namespace

universal::Span<const RegisteredCapabilityRecord<hiko_u::PackageInputKind>>
input_kinds_registry() noexcept {
  static const RegisteredCapabilityRecord<hiko_u::PackageInputKind> kRecords[] = {
      {hiko_u::PackageInputKind::StructureBackboneAtoms,
       "structure_backbone_atoms", true},
      {hiko_u::PackageInputKind::CoordsBackbone, "coords_backbone", true},
      {hiko_u::PackageInputKind::ResidueEmbeddings, "residue_embeddings", true},
      {hiko_u::PackageInputKind::StructureAllAtom, "structure_all_atom", false},
      // SequenceTokens is implemented now that the ESM2-8M architecture
      // accepts the route. Forward-pass execution still depends on a
      // populated architecture builder, but the descriptor surface and
      // tokenizer table are live in Hikoboshi 0.1.0.
      {hiko_u::PackageInputKind::SequenceTokens, "sequence_tokens", true},
      {hiko_u::PackageInputKind::DirectScoreMatrix, "direct_score_matrix", false},
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

universal::Span<const RegisteredCapabilityRecord<hiko_u::PackageOutputKind>>
output_kinds_registry() noexcept {
  static const RegisteredCapabilityRecord<hiko_u::PackageOutputKind> kRecords[] = {
      {hiko_u::PackageOutputKind::ResidueEmbeddings, "residue_embeddings", true},
      {hiko_u::PackageOutputKind::SubstitutionScores, "substitution_scores",
       false},
      {hiko_u::PackageOutputKind::DirectPairScores, "direct_pair_scores", false},
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

universal::Span<
    const RegisteredCapabilityRecord<hiko_u::PackagePreprocessingCapability>>
preprocessing_kinds_registry() noexcept {
  static const RegisteredCapabilityRecord<hiko_u::PackagePreprocessingCapability>
      kRecords[] = {
          {hiko_u::PackagePreprocessingCapability::AtomInference, "atom_inference",
           true},
          {hiko_u::PackagePreprocessingCapability::VirtualCb, "virtual_cb", true},
          {hiko_u::PackagePreprocessingCapability::CaKnn, "ca_knn", true},
          {hiko_u::PackagePreprocessingCapability::AtomPairDistances,
           "atom_pair_distances", true},
          {hiko_u::PackagePreprocessingCapability::RbfExpand, "rbf_expand", true},
          {hiko_u::PackagePreprocessingCapability::PositionalEncoding,
           "positional_encoding", true},
          // Tokenization is implemented now that the ESM2-8M architecture
          // ships a concrete 29-token table in embedded_esm2_8m.cpp
          // (Casey's compacted local alphabet).
          {hiko_u::PackagePreprocessingCapability::Tokenization, "tokenization",
           true},
      };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

universal::Span<const RegisteredCapabilityRecord<hiko_u::DataType>>
dtypes_registry() noexcept {
  static const RegisteredCapabilityRecord<hiko_u::DataType> kRecords[] = {
      {hiko_u::DataType::Float32, "float32", true},
      {hiko_u::DataType::Float64, "float64", false},
      {hiko_u::DataType::Int32, "int32", false},
      {hiko_u::DataType::UInt8, "uint8", false},
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

universal::Span<const RegisteredCapabilityRecord<hiko_u::PackageTensorLayout>>
layouts_registry() noexcept {
  static const RegisteredCapabilityRecord<hiko_u::PackageTensorLayout>
      kRecords[] = {
          {hiko_u::PackageTensorLayout::RowMajor, "row_major", true},
      };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

universal::Span<const RegisteredCapabilityRecord<DeviceKind>>
devices_registry() noexcept {
  static const RegisteredCapabilityRecord<DeviceKind> kRecords[] = {
      {DeviceKind::Cpu, "cpu", true},
      {DeviceKind::Gpu, "gpu", false},
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

universal::Span<
    const RegisteredCapabilityRecord<hiko_u::PackageBackendRequirement>>
backends_registry() noexcept {
  static const RegisteredCapabilityRecord<hiko_u::PackageBackendRequirement>
      kRecords[] = {
          {hiko_u::PackageBackendRequirement::CpuScalar, "cpu.scalar", true},
          {hiko_u::PackageBackendRequirement::GpuCuda, "gpu.cuda", false},
          {hiko_u::PackageBackendRequirement::GpuMetal, "gpu.metal", false},
          {hiko_u::PackageBackendRequirement::GpuHip, "gpu.hip", false},
      };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

}  // namespace hikoboshi::dispatch::registry
