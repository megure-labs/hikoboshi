#include <hikoboshi/modules/esm2.hpp>

#include <hikoboshi/universal/span.hpp>

#include <cstddef>

namespace hikoboshi::modules {
namespace {

namespace hiko_u = hikoboshi::universal;

bool span_has_capacity(const hiko_u::Span<float>& span,
                       std::size_t required) noexcept {
  return span.data != nullptr && span.size >= required;
}

hiko_u::Status invalid(const char* detail) noexcept {
  return hiko_u::invalid_argument_status(detail);
}

hiko_u::Status failed_precondition(const char* detail) noexcept {
  return hiko_u::failed_precondition_status(detail);
}

hiko_u::Status validate_descriptor(const Esm2Descriptor& descriptor) noexcept {
  if (descriptor.hidden_dimension == 0U) {
    return invalid("ESM2 descriptor hidden_dimension must be non-zero");
  }
  if (descriptor.head_count == 0U || descriptor.head_dim == 0U) {
    return invalid("ESM2 descriptor head_count and head_dim must be non-zero");
  }
  if (descriptor.head_count * descriptor.head_dim != descriptor.hidden_dimension) {
    return invalid(
        "ESM2 descriptor head_count * head_dim must equal hidden_dimension");
  }
  if (descriptor.head_dim % 2U != 0U) {
    return invalid("ESM2 descriptor head_dim must be even (RoPE pairing)");
  }
  if (descriptor.ffn_hidden_dimension == 0U) {
    return invalid("ESM2 descriptor ffn_hidden_dimension must be non-zero");
  }
  if (descriptor.vocab_size == 0U) {
    return invalid("ESM2 descriptor vocab_size must be non-zero");
  }
  return hiko_u::ok_status();
}

hiko_u::Status validate_weights(const Esm2Descriptor& descriptor,
                             const detail::Esm2Weights& weights) noexcept {
  if (weights.embedding_table == nullptr) {
    return invalid("ESM2 embedding table pointer is null");
  }
  if (weights.vocab_size != descriptor.vocab_size ||
      weights.hidden_dimension != descriptor.hidden_dimension) {
    return invalid("ESM2 weights vocab/hidden disagree with descriptor");
  }
  if (descriptor.layer_count > 0U && weights.layers == nullptr) {
    return invalid("ESM2 layer weights pointer is null");
  }
  if (weights.final_norm.gamma == nullptr ||
      weights.final_norm.beta == nullptr ||
      weights.final_norm.dim != descriptor.hidden_dimension) {
    return invalid("ESM2 final_norm weights must cover hidden_dimension");
  }
  for (std::size_t layer = 0; layer < descriptor.layer_count; ++layer) {
    const detail::Esm2LayerWeights& lw = weights.layers[layer];
    if (lw.attn_pre_norm.gamma == nullptr || lw.attn_pre_norm.beta == nullptr ||
        lw.attn_pre_norm.dim != descriptor.hidden_dimension) {
      return invalid("ESM2 attention pre_norm weights are invalid");
    }
    if (lw.ffn_pre_norm.gamma == nullptr || lw.ffn_pre_norm.beta == nullptr ||
        lw.ffn_pre_norm.dim != descriptor.hidden_dimension) {
      return invalid("ESM2 FFN pre_norm weights are invalid");
    }
    if (lw.wq.weight == nullptr || lw.wk.weight == nullptr ||
        lw.wv.weight == nullptr || lw.wo.weight == nullptr) {
      return invalid("ESM2 attention projection weights are null");
    }
    if (lw.wq.output_dim != descriptor.hidden_dimension ||
        lw.wq.input_dim != descriptor.hidden_dimension ||
        lw.wk.output_dim != descriptor.hidden_dimension ||
        lw.wk.input_dim != descriptor.hidden_dimension ||
        lw.wv.output_dim != descriptor.hidden_dimension ||
        lw.wv.input_dim != descriptor.hidden_dimension ||
        lw.wo.output_dim != descriptor.hidden_dimension ||
        lw.wo.input_dim != descriptor.hidden_dimension) {
      return invalid("ESM2 attention projection shapes must be [hidden, hidden]");
    }
    if (lw.ffn_in.weight == nullptr || lw.ffn_out.weight == nullptr) {
      return invalid("ESM2 FFN projection weights are null");
    }
    if (lw.ffn_in.output_dim != descriptor.ffn_hidden_dimension ||
        lw.ffn_in.input_dim != descriptor.hidden_dimension ||
        lw.ffn_out.output_dim != descriptor.hidden_dimension ||
        lw.ffn_out.input_dim != descriptor.ffn_hidden_dimension) {
      return invalid(
          "ESM2 FFN projection shapes must be [ffn_hidden, hidden] / "
          "[hidden, ffn_hidden]");
    }
  }
  return hiko_u::ok_status();
}

hiko_u::Status workspace_matches(const Esm2Descriptor& descriptor,
                              std::size_t seq_len,
                              const detail::Esm2Workspace& workspace) noexcept {
  const detail::Esm2MemoryPlan& plan = workspace.plan;
  if (plan.max_seq_len < seq_len ||
      plan.hidden_dim != descriptor.hidden_dimension ||
      plan.head_count != descriptor.head_count ||
      plan.head_dim != descriptor.head_dim ||
      plan.ffn_hidden_dim != descriptor.ffn_hidden_dimension) {
    return failed_precondition("ESM2 workspace plan disagrees with descriptor");
  }
  const std::size_t hidden_count = detail::esm2_hidden_state_count(plan);
  const std::size_t rope_count = detail::esm2_rope_table_count(plan);
  if (!span_has_capacity(workspace.hidden_state, hidden_count) ||
      !span_has_capacity(workspace.hidden_state_post_attn, hidden_count) ||
      !span_has_capacity(workspace.ffn_norm_buffer, hidden_count) ||
      !span_has_capacity(workspace.ffn_residual_buffer, hidden_count)) {
    return failed_precondition(
        "ESM2 workspace hidden-state buffers do not cover the plan");
  }
  if (!span_has_capacity(workspace.rope_cos, rope_count) ||
      !span_has_capacity(workspace.rope_sin, rope_count)) {
    return failed_precondition(
        "ESM2 workspace RoPE tables do not cover the plan");
  }
  if (workspace.attention_workspace.plan.max_seq_len < seq_len ||
      workspace.attention_workspace.plan.hidden_dim != descriptor.hidden_dimension ||
      workspace.attention_workspace.plan.head_count != descriptor.head_count ||
      workspace.attention_workspace.plan.head_dim != descriptor.head_dim) {
    return failed_precondition(
        "ESM2 attention sub-workspace plan disagrees with descriptor");
  }
  if (workspace.ffn_workspace.intermediate_buffer == nullptr ||
      workspace.ffn_workspace.intermediate_capacity <
          detail::esm2_ffn_intermediate_count(plan)) {
    return failed_precondition(
        "ESM2 FFN sub-workspace intermediate buffer is too small");
  }
  return hiko_u::ok_status();
}

}  // namespace

namespace detail {
universal::Status esm2_forward_scalar_unchecked(
    const Esm2ForwardRequest& request,
    const Esm2ForwardOutput& output) noexcept;
}  // namespace detail

universal::Status esm2_forward_scalar(
    const Esm2ForwardRequest& request,
    const Esm2ForwardOutput& output) noexcept {
  if (request.token_ids == nullptr) {
    return invalid("ESM2 token_ids pointer is null");
  }
  if (request.weights == nullptr) {
    return invalid("ESM2 weights pointer is null");
  }
  if (request.workspace == nullptr) {
    return invalid("ESM2 workspace pointer is null");
  }
  if (output.embeddings == nullptr) {
    return invalid("ESM2 output embedding pointer is null");
  }
  hiko_u::Status status = validate_descriptor(request.descriptor);
  if (!hiko_u::is_ok(status)) {
    return status;
  }
  if (output.hidden_dimension != request.descriptor.hidden_dimension) {
    return invalid("ESM2 output hidden_dimension disagrees with descriptor");
  }
  if (request.seq_len > request.descriptor.max_sequence_length) {
    return invalid("ESM2 request seq_len exceeds descriptor max_sequence_length");
  }
  if (output.seq_len < request.seq_len) {
    return invalid("ESM2 output seq_len capacity is smaller than input");
  }
  status = validate_weights(request.descriptor, *request.weights);
  if (!hiko_u::is_ok(status)) {
    return status;
  }
  status = workspace_matches(request.descriptor, request.seq_len,
                             *request.workspace);
  if (!hiko_u::is_ok(status)) {
    return status;
  }
  if (request.seq_len == 0U) {
    return hiko_u::ok_status();
  }
  return detail::esm2_forward_scalar_unchecked(request, output);
}

}  // namespace hikoboshi::modules
