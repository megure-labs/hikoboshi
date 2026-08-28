#ifndef HIKOBOSHI_UNIVERSAL_TENSOR_ROLE_HPP
#define HIKOBOSHI_UNIVERSAL_TENSOR_ROLE_HPP

/// @file
/// Closed tensor and parameter role vocabularies shared by package descriptors
/// and primitive-op registry records.
///
/// `TensorRole` is the closed vocabulary that describes both stored package
/// tensors and the runtime tensors that flow through primitive ops. The
/// stored-tensor values mirror the `tensor_role` enumeration from the
/// 2026-04-26 model-package execution schema decision report; the runtime
/// values describe the kind of tensor a primitive op consumes or produces so
/// graph-IR consumers and validators can reason about op signatures.
///
/// `ParameterRole` names the scalar parameters carried alongside a primitive
/// op (`k` for KNN, `sigma` for RBF, gap parameters for Smith-Waterman, …).

#include <cstdint>

namespace hikoboshi::universal {

/// Role of a tensor input, output, or stored payload.
///
/// The first block (up to and including `HistoricalOnly`) is the package
/// storage vocabulary that the architecture decision report enumerates. The
/// remaining values are runtime roles for primitive-op inputs and outputs.
///
/// Adding values is additive per the decision report's additivity rule;
/// changing existing values would require a `package_schema_version` bump.
enum class TensorRole : std::uint8_t {
  // Package storage roles (architecture decision report `tensor_role`).
  EncoderWeight = 0,
  EncoderBias = 1,
  EncoderNormGamma = 2,
  EncoderNormBeta = 3,
  EmbeddingTable = 4,
  PositionalTable = 5,
  SubstitutionMatrix = 6,
  GapParameterRecord = 7,
  HistoricalOnly = 8,
  // Runtime tensor roles for primitive op signatures.
  Coordinates3d = 16,
  AtomSource = 17,
  ValidityMask = 18,
  NeighborIndices = 19,
  GatherIndices = 20,
  SquaredDistances = 21,
  RbfFeatures = 22,
  ResidueEmbeddings = 23,
  ActivationMatrix = 24,
  ReducedVector = 25,
  ScoreMatrix = 26,
  AlignmentPath = 27,
};

/// Role of a scalar parameter carried beside primitive op inputs and outputs.
///
/// Parameters are non-tensor scalar arguments captured in the op's request
/// struct (`KnnScalarRequest::k`, `RbfScalarRequest::sigma`, etc.). The
/// vocabulary stays closed so graph-IR validators can match against it.
enum class ParameterRole : std::uint8_t {
  NeighborCount = 0,
  IncludeSelfFlag = 1,
  TreatZeroAsInvalidFlag = 2,
  FeatureCount = 3,
  CenterMin = 4,
  CenterMax = 5,
  Sigma = 6,
  Epsilon = 7,
  Alpha = 8,
  Beta = 9,
  RowCount = 10,
  ColumnCount = 11,
  InnerDim = 12,
  GapOpen = 13,
  GapExtension = 14,
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_TENSOR_ROLE_HPP
