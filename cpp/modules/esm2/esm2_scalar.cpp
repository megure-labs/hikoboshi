#include <hikoboshi/modules/esm2.hpp>

#include <hikoboshi/dispatch/scalar_forward.hpp>
#include <hikoboshi/modules/ffn/detail/ffn_layer_inline.hpp>
#include <hikoboshi/modules/ffn/ffn_layer.hpp>
#include <hikoboshi/modules/transformer/detail/attention_inline.hpp>

#include <cstddef>
#include <cstring>

namespace hikoboshi::modules {
namespace {

namespace hiko_u = hikoboshi::universal;
namespace hiko_p = hikoboshi::primitives;
namespace hiko_d = hikoboshi::dispatch;
namespace hiko_td = hikoboshi::modules::transformer::detail;
namespace hiko_fd = hikoboshi::modules::ffn::detail;

// Copy `[seq_len, hidden_dim]` rows from the embedding table indexed by the
// token id buffer. Token ids out of range collapse to row 0; the engine-side
// validator rejects out-of-range tokens before reaching this function, but
// the bounds check stays here as a defense-in-depth measure so a corrupt
// blob cannot dereference past the embedding table.
void embedding_lookup(const std::int32_t* token_ids,
                      const float* embedding_table, std::size_t seq_len,
                      std::size_t vocab_size, std::size_t hidden_dim,
                      float* hidden_state) noexcept {
  for (std::size_t s = 0; s < seq_len; ++s) {
    const std::int32_t raw = token_ids[s];
    std::size_t row = 0U;
    if (raw >= 0 && static_cast<std::size_t>(raw) < vocab_size) {
      row = static_cast<std::size_t>(raw);
    }
    const float* source = embedding_table + row * hidden_dim;
    float* destination = hidden_state + s * hidden_dim;
    std::memcpy(destination, source, hidden_dim * sizeof(float));
  }
}

void apply_attention_block(const detail::Esm2LayerWeights& weights,
                           std::size_t seq_len, std::size_t hidden_dim,
                           std::size_t head_count, std::size_t head_dim,
                           const float* rope_cos, const float* rope_sin,
                           hiko_td::TransformerAttentionWorkspace& workspace,
                           const float* input, float* output) noexcept {
  hikoboshi::modules::transformer::AttentionLayerRequest request{};
  request.input_embeddings = input;
  request.seq_len = seq_len;
  request.hidden_dim = hidden_dim;
  request.head_count = head_count;
  request.head_dim = head_dim;
  request.wq = weights.wq;
  request.wk = weights.wk;
  request.wv = weights.wv;
  request.wo = weights.wo;
  request.pre_norm = weights.attn_pre_norm;
  request.rope_cos = rope_cos;
  request.rope_sin = rope_sin;
  request.attention_mask = nullptr;
  request.workspace = &workspace;
  hikoboshi::modules::transformer::AttentionLayerOutput out{};
  out.output_embeddings = output;
  hiko_td::attention_layer_scalar_inline(request, out);
}

void apply_ffn_block(const detail::Esm2LayerWeights& weights,
                     std::size_t seq_len, std::size_t hidden_dim,
                     std::size_t ffn_hidden_dim,
                     hiko_fd::FfnLayerWorkspace& workspace, float* ffn_norm_scratch,
                     const float* input, float* pre_residual,
                     float* output) noexcept {
  // Pre-FFN LayerNorm (the ESM2 block is pre-norm; the FFN template ships
  // with `NoNormTag`, so the encoder applies the norm itself before handing
  // the activation to the FFN body).
  hiko_p::compute::LayerNormScalarRequest norm_request{};
  norm_request.input = input;
  norm_request.gamma = weights.ffn_pre_norm.gamma;
  norm_request.beta = weights.ffn_pre_norm.beta;
  norm_request.row_count = seq_len;
  norm_request.row_dimension = hidden_dim;
  norm_request.epsilon = weights.ffn_pre_norm.epsilon;
  hiko_d::layer_norm_forward(hiko_d::ScalarTag{}, norm_request, ffn_norm_scratch);

  hikoboshi::modules::ffn::FfnLayerRequest request{};
  request.input_embeddings = ffn_norm_scratch;
  request.rows = seq_len;
  request.hidden_dim = hidden_dim;
  request.intermediate_dim = ffn_hidden_dim;
  request.w_in = weights.ffn_in;
  request.w_out = weights.ffn_out;
  request.workspace = &workspace;
  hikoboshi::modules::ffn::FfnLayerOutput ffn_out{};
  ffn_out.output_embeddings = pre_residual;
  hiko_fd::ffn_layer_inline<hikoboshi::modules::ffn::GeluTag,
                         hikoboshi::modules::ffn::NoNormTag,
                         hikoboshi::modules::ffn::NoResidualTag,
                         /*HasBias=*/true, hiko_d::FastParityTag,
                         hiko_d::ScalarTag>(request, ffn_out);

  // Residual add: output = input + pre_residual. Routes through the
  // primitive-level axpy so the registered primitive list stays the
  // closed surface the encoder composes.
  hiko_p::compute::AxpyScalarRequest axpy_request{};
  axpy_request.x = pre_residual;
  axpy_request.y = input;
  axpy_request.alpha = 1.0F;
  axpy_request.count = seq_len * hidden_dim;
  hiko_d::axpy_forward(hiko_d::ScalarTag{}, axpy_request, output);
}

void apply_final_norm(const detail::Esm2Weights& weights, std::size_t seq_len,
                      std::size_t hidden_dim, const float* input,
                      float* output) noexcept {
  hiko_p::compute::LayerNormScalarRequest request{};
  request.input = input;
  request.gamma = weights.final_norm.gamma;
  request.beta = weights.final_norm.beta;
  request.row_count = seq_len;
  request.row_dimension = hidden_dim;
  request.epsilon = weights.final_norm.epsilon;
  hiko_d::layer_norm_forward(hiko_d::ScalarTag{}, request, output);
}

}  // namespace

namespace detail {

universal::Status esm2_forward_scalar_unchecked(
    const Esm2ForwardRequest& request,
    const Esm2ForwardOutput& output) noexcept {
  const Esm2Descriptor& descriptor = request.descriptor;
  const detail::Esm2Weights& weights = *request.weights;
  detail::Esm2Workspace& workspace = *request.workspace;
  const std::size_t seq_len = request.seq_len;
  const std::size_t hidden_dim = descriptor.hidden_dimension;
  const std::size_t head_count = descriptor.head_count;
  const std::size_t head_dim = descriptor.head_dim;
  const std::size_t ffn_hidden_dim = descriptor.ffn_hidden_dimension;

  embedding_lookup(request.token_ids, weights.embedding_table, seq_len,
                   descriptor.vocab_size, hidden_dim,
                   workspace.hidden_state.data);

  for (std::size_t layer = 0; layer < descriptor.layer_count; ++layer) {
    const detail::Esm2LayerWeights& layer_weights = weights.layers[layer];

    // Attention block: pre-norm + Q/K/V + RoPE + scaled-dot + W_O + residual.
    // The attention compound module owns the pre-norm and residual add
    // internally, so the encoder just hands it the embedding before the
    // residual and reads the post-residual buffer.
    apply_attention_block(layer_weights, seq_len, hidden_dim, head_count,
                          head_dim, workspace.rope_cos.data,
                          workspace.rope_sin.data, workspace.attention_workspace,
                          workspace.hidden_state.data,
                          workspace.hidden_state_post_attn.data);

    // FFN block: pre-norm + linear-up + GELU + linear-down + residual.
    // The FFN template ships with `NoNormTag`/`NoResidualTag`, so the
    // encoder applies the norm into a scratch buffer and the residual
    // add into the original hidden state.
    apply_ffn_block(layer_weights, seq_len, hidden_dim, ffn_hidden_dim,
                    workspace.ffn_workspace, workspace.ffn_norm_buffer.data,
                    workspace.hidden_state_post_attn.data,
                    workspace.ffn_residual_buffer.data,
                    workspace.hidden_state.data);
  }

  apply_final_norm(weights, seq_len, hidden_dim, workspace.hidden_state.data,
                   output.embeddings);

  return universal::ok_status();
}

}  // namespace detail

}  // namespace hikoboshi::modules
