#include <hikoboshi/dispatch/registry/module_op.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <hikoboshi/dispatch/backend_tag.hpp>
#include <hikoboshi/dispatch/registry/primitive_op.hpp>
#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/universal/package.hpp>
#include <hikoboshi/universal/span.hpp>
#include <hikoboshi/universal/status.hpp>
#include <hikoboshi/universal/tensor_role.hpp>

// Forward declarations for the compound-module symbols whose addresses the
// module-op records capture. The dispatch layer must not include
// `hikoboshi/modules/...` headers (the dependency graph allows
// `modules -> dispatch` only), so the records reference module functions as
// raw `const void*` function pointers via forward declarations of the module
// request/output structs and entry functions.
namespace hikoboshi::modules {

struct Mpnn64ForwardRequest;
struct Mpnn64ForwardOutput;

hikoboshi::universal::Status mpnn64_forward_scalar(
    const Mpnn64ForwardRequest& request,
    const Mpnn64ForwardOutput& output) noexcept;

struct Esm2ForwardRequest;
struct Esm2ForwardOutput;

hikoboshi::universal::Status esm2_forward_scalar(
    const Esm2ForwardRequest& request,
    const Esm2ForwardOutput& output) noexcept;

struct SimilarityScalarRequest;
struct SimilarityScalarOutput;

hikoboshi::universal::Status similarity_scalar(
    const SimilarityScalarRequest& request,
    const SimilarityScalarOutput& output) noexcept;

namespace mpnn {

struct MpnnMessageLayerRequest;
struct MpnnMessageLayerOutput;

void mpnn_message_layer_scalar(const MpnnMessageLayerRequest& request,
                               const MpnnMessageLayerOutput& output) noexcept;

struct MpnnFfnLayerRequest;
struct MpnnFfnLayerOutput;

void mpnn_ffn_layer_scalar(const MpnnFfnLayerRequest& request,
                           const MpnnFfnLayerOutput& output) noexcept;

}  // namespace mpnn

namespace transformer {

struct AttentionLayerRequest;
struct AttentionLayerOutput;

void attention_layer_scalar(const AttentionLayerRequest& request,
                            const AttentionLayerOutput& output) noexcept;

}  // namespace transformer

namespace ffn {

struct FfnLayerRequest;
struct FfnLayerOutput;

void ffn_layer_scalar_gelu_nonorm_noresidual_bias_fast(
    const FfnLayerRequest& request,
    const FfnLayerOutput& output) noexcept;

}  // namespace ffn

}  // namespace hikoboshi::modules

namespace hikoboshi::dispatch::registry {
namespace {

namespace hiko_u = hikoboshi::universal;
namespace hiko_p = hikoboshi::primitives;
namespace hiko_m = hikoboshi::modules;

// Entry-wrapper for the public hard-SW module-level contract. The
// module-level wrapper exists to distinguish the compound-module surface
// from the primitive-level kernel; in Hikoboshi 0.1.0 it delegates directly to
// the scalar primitive forward.
void hard_sw_wrapper_scalar_entry(
    const hiko_p::alignment::SmithWatermanScalarRequest& request,
    hiko_p::alignment::SmithWatermanScalarOutput& output) {
  smith_waterman_forward(ScalarTag{}, request, output);
}

// Family textual spellings. `kFamilyAttention` and `kFamilyFeedforward`
// distinguish the generic compounds (attention block, FFN template family)
// from the architecture-specific MPNN per-layer compounds that still spell
// their family as "layer"; all four still map to `ModuleOpFamily::Layer`
// at the enum level because the enum is the coarse role bucket used by
// graph-IR filtering, while the textual family is the diagnostic-facing
// spelling.
constexpr std::string_view kFamilyEncoder{"encoder"};
constexpr std::string_view kFamilySimilarity{"similarity"};
constexpr std::string_view kFamilyLayer{"layer"};
constexpr std::string_view kFamilyAttention{"attention"};
constexpr std::string_view kFamilyFeedforward{"feedforward"};
constexpr std::string_view kFamilyAlignmentWrapper{"alignment_wrapper"};

constexpr std::string_view kVersionV1{"v1"};

// Module-op identity strings. The dotted form is `hikoboshi.<area>.v1.<role>`
// for architecture-specific compounds (MPNN message_layer, MPNN ffn_layer,
// MPNN encoder) and `hikoboshi.<role>.v1` for generic compounds that any
// architecture composes (attention, ffn_layer). The similarity op keeps the
// `hikoboshi.<area>.<role>.v1` shape that matches the scoring registry's
// `dot_product_similarity_v1` tag.
constexpr std::string_view kOpIdMpnnEncoder{"hikoboshi.mpnn.v1.encoder"};
// ESM2-8M encoder compound. Composes the registered attention compound,
// the FFN template family, and the scalar LayerNorm primitive into a
// transformer-with-sequence-input encoder. Keyed in the
// `hikoboshi.<area>.v1.<role>` shape (matching the MPNN encoder) rather
// than the generic `hikoboshi.<role>.v1` form because every transformer-
// family encoder takes the same architecture-specific weights bundle
// and the registry naming exists to disambiguate those bundles.
constexpr std::string_view kOpIdEsm2Encoder{"hikoboshi.esm2.v1.encoder"};
constexpr std::string_view kOpIdSimilarityDotProduct{
    "hikoboshi.similarity.dot_product.v1"};
constexpr std::string_view kOpIdMpnnMessageLayer{
    "hikoboshi.mpnn.v1.message_layer"};
constexpr std::string_view kOpIdMpnnFfnLayer{"hikoboshi.mpnn.v1.ffn_layer"};
constexpr std::string_view kOpIdHardSwWrapper{
    "hikoboshi.alignment.hard_sw_wrapper.v1"};
// Generic attention compound. Re-keyed from `hikoboshi.transformer.attention.v1`
// by the compound-module-architecture-agnostic-interface packet because
// attention is composable by any attention-based architecture (transformer,
// ESM2, BERT, ESM3, …) and the closed-op-set additivity rule does not yet
// bind for pre-release op ids.
constexpr std::string_view kOpIdAttention{"hikoboshi.attention.v1"};
// Generic FFN template family. CUTLASS-style closed-tag-axis specialization
// surface; see `cpp/include/hikoboshi/modules/ffn/ffn_layer.hpp` for the
// template signature and `docs/charters/DSL_DESIGN_OVERVIEW.md` for the
// closed-tag-axis vs runtime-numeric specialization rationale.
constexpr std::string_view kOpIdFfnLayer{"hikoboshi.ffn_layer.v1"};

// Capability arrays. Hikoboshi 0.1.0 compound modules are wired only on the
// scalar CPU backend, and inherit the strict-only parity contract because
// the GEMM dual-mode axis is opt-in at the primitive boundary.
constexpr hiko_u::PackageBackendRequirement kScalarBackends[] = {
    hiko_u::PackageBackendRequirement::CpuScalar,
};

constexpr ParityMode kStrictOnly[] = {ParityMode::Strict};
constexpr ParityMode kStrictOrFast[] = {ParityMode::Strict, ParityMode::Fast};

// Required-primitive lists per module-op. Each entry is a canonical
// primitive op id that must resolve in `primitive_op_registry()` at
// cross-reference validation time.
constexpr std::string_view kMpnnEncoderRequiredPrimitives[] = {
    "hikoboshi.gemm.nt.v1",
    "hikoboshi.layer_norm.v1",
    "hikoboshi.gather.v1",
    "hikoboshi.rbf.v1",
    "hikoboshi.knn.v1",
    "hikoboshi.gelu.v1",
    "hikoboshi.bias_add.v1",
    "hikoboshi.axpy.v1",
};

// ESM2-8M encoder primitive list. The encoder composes the attention
// compound (which transitively requires layer_norm, both gemm variants,
// softmax, bias_add, and axpy) plus the FFN template family (gemm_nt,
// gelu, bias_add) plus its own pre-FFN LayerNorm and final-encoder
// LayerNorm. Each id below must resolve in `primitive_op_registry()`.
constexpr std::string_view kEsm2EncoderRequiredPrimitives[] = {
    "hikoboshi.layer_norm.v1",
    "hikoboshi.gemm.nt.v1",
    "hikoboshi.gemm.nn.v1",
    "hikoboshi.softmax.row_wise.v1",
    "hikoboshi.gelu.v1",
    "hikoboshi.bias_add.v1",
    "hikoboshi.axpy.v1",
};

constexpr std::string_view kSimilarityDotProductRequiredPrimitives[] = {
    "hikoboshi.gemm.nt.v1",
};

constexpr std::string_view kMpnnMessageLayerRequiredPrimitives[] = {
    "hikoboshi.gemm.nt.v1",
    "hikoboshi.layer_norm.v1",
    "hikoboshi.gelu.v1",
    "hikoboshi.bias_add.v1",
};

constexpr std::string_view kMpnnFfnLayerRequiredPrimitives[] = {
    "hikoboshi.gemm.nt.v1",
    "hikoboshi.layer_norm.v1",
    "hikoboshi.gelu.v1",
    "hikoboshi.bias_add.v1",
};

constexpr std::string_view kHardSwWrapperRequiredPrimitives[] = {
    "hikoboshi.smith_waterman.v1",
    "hikoboshi.traceback.v1",
};

constexpr std::string_view kTransformerAttentionRequiredPrimitives[] = {
    "hikoboshi.layer_norm.v1",
    "hikoboshi.gemm.nt.v1",
    "hikoboshi.gemm.nn.v1",
    "hikoboshi.softmax.row_wise.v1",
    "hikoboshi.bias_add.v1",
    "hikoboshi.axpy.v1",
};

// FFN template family. The closed tag tuple shipped in 0.1.0 is
// `(GeluTag, NoNormTag, NoResidualTag, HasBias=true)`, so the body composes
// two `gemm_nt` calls flanking an in-place GELU plus per-linear `bias_add`.
// Future tag tuples that need a norm or residual step will declare the
// extra primitive ids when their ABI wrappers ship.
constexpr std::string_view kFfnLayerRequiredPrimitives[] = {
    "hikoboshi.gemm.nt.v1",
    "hikoboshi.gelu.v1",
    "hikoboshi.bias_add.v1",
};

// Per-module-op tensor signature arrays.

constexpr hiko_u::TensorRole kMpnnEncoderInputs[] = {
    hiko_u::TensorRole::Coordinates3d,
    hiko_u::TensorRole::AtomSource,
};
constexpr hiko_u::TensorRole kMpnnEncoderOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};

// ESM2-8M encoder tensor signature. The closed `TensorRole` vocabulary
// has no dedicated `TokenIds` entry; sequence tokens semantically are
// indices that gather rows from the embedding table, so the input is
// tagged with `GatherIndices` to reuse the existing role rather than
// extending the enum from a worker packet. The embedding table, per-
// layer transformer weights, and final-norm gamma/beta share the
// `EmbeddingTable` / `EncoderWeight` / `EncoderBias` /
// `EncoderNormGamma` / `EncoderNormBeta` roles already declared on the
// transformer attention and FFN compounds. `PositionalTable` covers the
// precomputed RoPE cos/sin tables that attention reads each layer.
constexpr hiko_u::TensorRole kEsm2EncoderInputs[] = {
    hiko_u::TensorRole::GatherIndices,
    hiko_u::TensorRole::EmbeddingTable,
    hiko_u::TensorRole::EncoderWeight,
    hiko_u::TensorRole::EncoderBias,
    hiko_u::TensorRole::EncoderNormGamma,
    hiko_u::TensorRole::EncoderNormBeta,
    hiko_u::TensorRole::PositionalTable,
};
constexpr hiko_u::TensorRole kEsm2EncoderOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};

constexpr hiko_u::TensorRole kSimilarityDotProductInputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
    hiko_u::TensorRole::ResidueEmbeddings,
};
constexpr hiko_u::TensorRole kSimilarityDotProductOutputs[] = {
    hiko_u::TensorRole::ScoreMatrix,
};

constexpr hiko_u::TensorRole kMpnnMessageLayerInputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
    hiko_u::TensorRole::ActivationMatrix,
    hiko_u::TensorRole::NeighborIndices,
    hiko_u::TensorRole::EncoderWeight,
    hiko_u::TensorRole::EncoderBias,
    hiko_u::TensorRole::EncoderNormGamma,
    hiko_u::TensorRole::EncoderNormBeta,
};
constexpr hiko_u::TensorRole kMpnnMessageLayerOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};

constexpr hiko_u::TensorRole kMpnnFfnLayerInputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
    hiko_u::TensorRole::EncoderWeight,
    hiko_u::TensorRole::EncoderBias,
    hiko_u::TensorRole::EncoderNormGamma,
    hiko_u::TensorRole::EncoderNormBeta,
};
constexpr hiko_u::TensorRole kMpnnFfnLayerOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};

constexpr hiko_u::TensorRole kHardSwWrapperInputs[] = {
    hiko_u::TensorRole::ScoreMatrix,
    hiko_u::TensorRole::GapParameterRecord,
};
constexpr hiko_u::TensorRole kHardSwWrapperOutputs[] = {
    hiko_u::TensorRole::AlignmentPath,
};

// Attention compound module. Consumes a row-major sequence-embedding
// activation, the four projection weight/bias pairs (Q/K/V/O), and the
// pre-attention LayerNorm gamma/beta. Optional positional cos/sin tables
// are tagged with `PositionalTable`. The additive attention mask has no
// exact match in the closed `TensorRole` vocabulary (the same gap softmax
// records) and is therefore omitted from the signature; future vocabulary
// extensions can broaden it without changing the op identity.
constexpr hiko_u::TensorRole kAttentionInputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
    hiko_u::TensorRole::EncoderWeight,
    hiko_u::TensorRole::EncoderBias,
    hiko_u::TensorRole::EncoderNormGamma,
    hiko_u::TensorRole::EncoderNormBeta,
    hiko_u::TensorRole::PositionalTable,
};
constexpr hiko_u::TensorRole kAttentionOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};

// FFN template family. Consumes a row-major sequence-embedding activation
// plus two `LinearLayerWeightsView` weight/bias pairs (w_in projects up to
// the intermediate dim, w_out projects back to the hidden dim). The 0.1.0
// tuple uses `NoNormTag` and `NoResidualTag`, so the signature does not
// declare LayerNorm gamma/beta tensors; architectures that wrap norm or
// residual steps externally pass those through their own surfaces.
constexpr hiko_u::TensorRole kFfnLayerInputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
    hiko_u::TensorRole::EncoderWeight,
    hiko_u::TensorRole::EncoderBias,
};
constexpr hiko_u::TensorRole kFfnLayerOutputs[] = {
    hiko_u::TensorRole::ResidueEmbeddings,
};

template <typename T, std::size_t N>
constexpr hiko_u::Span<const T> as_span(const T (&array)[N]) noexcept {
  return {array, N};
}

constexpr std::string_view kDiagnosticCodeMissingPrimitive{
    "module_op_required_primitive_missing"};
constexpr std::string_view kDiagnosticMessageMissingPrimitive{
    "module-op required_primitive_op_ids entry does not resolve in "
    "primitive_op_registry()"};

// Internal accessor for the records storage. Both `module_op_registry()`
// (the public surface that also triggers cross-reference validation) and
// the cached-validation helper read records through this function so the
// validation helper can call us without recursing through the public
// surface.
universal::Span<const RegisteredModuleOpRecord>
module_op_records_internal() noexcept {
  static const RegisteredModuleOpRecord kRecords[] = {
      {
          {kOpIdMpnnEncoder, kFamilyEncoder, kVersionV1},
          ModuleOpFamily::Encoder,
          {as_span(kMpnnEncoderInputs), as_span(kMpnnEncoderOutputs),
           as_span(kMpnnEncoderRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&hiko_m::mpnn64_forward_scalar),
      },
      {
          {kOpIdSimilarityDotProduct, kFamilySimilarity, kVersionV1},
          ModuleOpFamily::Similarity,
          {as_span(kSimilarityDotProductInputs),
           as_span(kSimilarityDotProductOutputs),
           as_span(kSimilarityDotProductRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&hiko_m::similarity_scalar),
      },
      {
          {kOpIdMpnnMessageLayer, kFamilyLayer, kVersionV1},
          ModuleOpFamily::Layer,
          {as_span(kMpnnMessageLayerInputs),
           as_span(kMpnnMessageLayerOutputs),
           as_span(kMpnnMessageLayerRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&hiko_m::mpnn::mpnn_message_layer_scalar),
      },
      {
          {kOpIdMpnnFfnLayer, kFamilyLayer, kVersionV1},
          ModuleOpFamily::Layer,
          {as_span(kMpnnFfnLayerInputs), as_span(kMpnnFfnLayerOutputs),
           as_span(kMpnnFfnLayerRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&hiko_m::mpnn::mpnn_ffn_layer_scalar),
      },
      {
          {kOpIdHardSwWrapper, kFamilyAlignmentWrapper, kVersionV1},
          ModuleOpFamily::AlignmentWrapper,
          {as_span(kHardSwWrapperInputs), as_span(kHardSwWrapperOutputs),
           as_span(kHardSwWrapperRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOnly), ParityMode::Strict},
          reinterpret_cast<const void*>(&hard_sw_wrapper_scalar_entry),
      },
      // Generic multi-head scaled-dot-product attention compound module.
      // Hikoboshi 0.1.0 scalar backend only. Strict and Fast parity modes
      // are both supported (Fast routes the internal GEMMs through the
      // BLIS-style fast kernel selected by `HIKOBOSHI_GEMM_PARITY_MODE`);
      // the registry's default parity for this op is Fast because the
      // per-layer attention block is a stacked-GEMM hot path that benefits
      // from the fast kernel out of the box, while the strict mode remains
      // available for parity goldens. The textual op_family spells
      // "attention" rather than "layer" because attention is a generic
      // compound consumed by any attention-based architecture; the
      // `ModuleOpFamily::Layer` enum value continues to bucket it into the
      // coarse per-layer role for graph-IR filtering.
      {
          {kOpIdAttention, kFamilyAttention, kVersionV1},
          ModuleOpFamily::Layer,
          {as_span(kAttentionInputs), as_span(kAttentionOutputs),
           as_span(kTransformerAttentionRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOrFast), ParityMode::Fast},
          reinterpret_cast<const void*>(
              &hiko_m::transformer::attention_layer_scalar),
      },
      // FFN template family. CUTLASS-style closed-tag-axis specialization
      // surface; Hikoboshi 0.1.0 ships the
      // `(GeluTag, NoNormTag, NoResidualTag, HasBias=true)` tuple in both
      // strict and fast parity flavors. The dispatch entry below points at
      // the fast-parity ABI wrapper; the strict-parity wrapper is reachable
      // by name for parity-golden tests. The textual op_family spells
      // "feedforward" rather than "layer" to mirror the attention rename;
      // the `ModuleOpFamily::Layer` enum value continues to bucket it into
      // the coarse per-layer role.
      {
          {kOpIdFfnLayer, kFamilyFeedforward, kVersionV1},
          ModuleOpFamily::Layer,
          {as_span(kFfnLayerInputs), as_span(kFfnLayerOutputs),
           as_span(kFfnLayerRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOrFast), ParityMode::Fast},
          reinterpret_cast<const void*>(
              &hiko_m::ffn::ffn_layer_scalar_gelu_nonorm_noresidual_bias_fast),
      },
      // ESM2-8M encoder compound. Hikoboshi 0.1.0 ships only the scalar
      // backend; the encoder composes the registered attention compound,
      // the FFN template family, and the scalar LayerNorm primitive.
      // Strict and Fast parity modes are both supported so parity goldens
      // can pin the strict path while production runs benefit from the
      // BLIS-style fast GEMM kernel through the attention/FFN sub-blocks.
      // The default parity is Fast for production runs; per-call control
      // lives in the underlying GEMM tag selection.
      {
          {kOpIdEsm2Encoder, kFamilyEncoder, kVersionV1},
          ModuleOpFamily::Encoder,
          {as_span(kEsm2EncoderInputs), as_span(kEsm2EncoderOutputs),
           as_span(kEsm2EncoderRequiredPrimitives)},
          {as_span(kScalarBackends), as_span(kStrictOrFast), ParityMode::Fast},
          reinterpret_cast<const void*>(&hiko_m::esm2_forward_scalar),
      },
  };
  return {kRecords, sizeof(kRecords) / sizeof(kRecords[0])};
}

// Internal: walk a single module-op record's required primitive ids and
// append a diagnostic for each id that does not resolve in
// `primitive_op_registry()`. Returns the number of diagnostics written
// (including those truncated past `buffer_capacity`, so callers can detect
// truncation by comparing against `buffer_capacity`).
std::size_t validate_one_record(
    const RegisteredModuleOpRecord& record,
    ModuleOpValidationDiagnostic* diagnostic_buffer,
    std::size_t buffer_capacity, std::size_t starting_index) noexcept {
  std::size_t written_or_truncated = 0;
  const hiko_u::Span<const std::string_view> required =
      record.signature.required_primitive_op_ids;
  for (std::size_t r = 0; r < required.size; ++r) {
    const std::string_view required_id = required.data[r];
    if (find_primitive_op(required_id) != nullptr) {
      continue;
    }
    const std::size_t slot = starting_index + written_or_truncated;
    if (slot < buffer_capacity && diagnostic_buffer != nullptr) {
      diagnostic_buffer[slot] = ModuleOpValidationDiagnostic{
          ModuleOpValidationSeverity::Error,
          kDiagnosticCodeMissingPrimitive,
          record.identity.op_id,
          required_id,
          kDiagnosticMessageMissingPrimitive,
      };
    }
    ++written_or_truncated;
  }
  return written_or_truncated;
}

// Bounded static buffer for the cached live-registry validation report.
// Sized to the worst case of summing every record's required-primitive
// count (8 module-ops with at most 8 required primitives each). 96 keeps
// headroom for additive growth without inflating BSS.
constexpr std::size_t kLiveValidationCapacity = 96;

}  // namespace

ModuleOpValidationReport validate_module_op_records(
    const universal::Span<const RegisteredModuleOpRecord> records,
    ModuleOpValidationDiagnostic* diagnostic_buffer,
    const std::size_t buffer_capacity) noexcept {
  std::size_t total_emitted = 0;
  bool ok = true;
  for (std::size_t i = 0; i < records.size; ++i) {
    const std::size_t emitted = validate_one_record(
        records.data[i], diagnostic_buffer, buffer_capacity, total_emitted);
    if (emitted > 0) {
      ok = false;
    }
    total_emitted += emitted;
  }
  const std::size_t fit_count =
      total_emitted <= buffer_capacity ? total_emitted : buffer_capacity;
  return ModuleOpValidationReport{
      {diagnostic_buffer, fit_count},
      ok,
  };
}

const ModuleOpValidationReport& module_op_registry_validation() noexcept {
  static ModuleOpValidationDiagnostic kBuffer[kLiveValidationCapacity];
  static const ModuleOpValidationReport kReport = validate_module_op_records(
      module_op_records_internal(), kBuffer, kLiveValidationCapacity);
  return kReport;
}

universal::Span<const RegisteredModuleOpRecord>
module_op_registry() noexcept {
  // Trigger first-access cross-reference validation as a side effect so
  // dependent consumers (validation-pipeline-completion) can rely on the
  // cached report when they walk the registry. Validation reads records
  // through `module_op_records_internal()` to avoid recursing back into
  // this public surface during static initialization.
  (void)module_op_registry_validation();
  return module_op_records_internal();
}

const RegisteredModuleOpRecord* find_module_op(
    const std::string_view op_id) noexcept {
  const universal::Span<const RegisteredModuleOpRecord> records =
      module_op_registry();
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].identity.op_id == op_id) {
      return &records.data[index];
    }
  }
  return nullptr;
}

}  // namespace hikoboshi::dispatch::registry
