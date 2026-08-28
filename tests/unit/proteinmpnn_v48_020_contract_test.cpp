#include <hikoboshi/primitives/compute/log_softmax.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_schema.hpp>
#include <hikoboshi/universal/detail/proteinmpnn_v48_020_weights.hpp>

#include <array>
#include <cstddef>
#include <string_view>
#include <type_traits>

namespace hiko_p = hikoboshi::primitives::compute;
namespace hiko_s = hikoboshi::universal::detail;

static_assert(hiko_s::ProteinMpnnV48020Weights::hidden == 128,
              "ProteinMPNN v_48_020 hidden dimension mismatch");
static_assert(hiko_s::ProteinMpnnV48020Weights::k_neighbors == 48,
              "ProteinMPNN v_48_020 neighbor count mismatch");
static_assert(hiko_s::ProteinMpnnV48020Weights::vocab == 21,
              "ProteinMPNN v_48_020 vocab size mismatch");
static_assert(hiko_s::ProteinMpnnV48020Weights::rbf_count == 16,
              "ProteinMPNN v_48_020 RBF count mismatch");
static_assert(hiko_s::ProteinMpnnV48020Weights::num_encoder_layers == 3,
              "ProteinMPNN v_48_020 encoder layer count mismatch");
static_assert(hiko_s::ProteinMpnnV48020Weights::num_decoder_layers == 3,
              "ProteinMPNN v_48_020 decoder layer count mismatch");
static_assert(hiko_s::ProteinMpnnV48020Weights::message_scale == 30.0F,
              "ProteinMPNN v_48_020 message scale mismatch");
static_assert(hiko_s::ProteinMpnnV48020EncoderLayerWeights::tensor_count == 22,
              "ProteinMPNN v_48_020 encoder layer field count mismatch");
static_assert(hiko_s::ProteinMpnnV48020DecoderLayerWeights::tensor_count == 14,
              "ProteinMPNN v_48_020 decoder layer field count mismatch");
static_assert(hiko_s::ProteinMpnnV48020Weights::tensor_count == 118,
              "ProteinMPNN v_48_020 weight view field count mismatch");
static_assert(hiko_s::kProteinMpnnV48020SchemaTensorCount == 118,
              "ProteinMPNN v_48_020 schema tensor count mismatch");

static_assert(
    std::is_same<decltype(hiko_p::log_softmax_scalar),
                 void(const hiko_p::LogSoftmaxScalarRequest&,
                      const hiko_p::LogSoftmaxScalarOutput&)>::value,
    "log_softmax scalar signature mismatch");

template <typename T, typename = void>
struct HasW11 : std::false_type {};
template <typename T>
struct HasW11<T, std::void_t<decltype(&T::W11)>> : std::true_type {};

template <typename T, typename = void>
struct HasW12 : std::false_type {};
template <typename T>
struct HasW12<T, std::void_t<decltype(&T::W12)>> : std::true_type {};

template <typename T, typename = void>
struct HasW13 : std::false_type {};
template <typename T>
struct HasW13<T, std::void_t<decltype(&T::W13)>> : std::true_type {};

template <typename T, typename = void>
struct HasNorm3 : std::false_type {};
template <typename T>
struct HasNorm3<T, std::void_t<decltype(&T::norm3)>> : std::true_type {};

static_assert(HasW11<hiko_s::ProteinMpnnV48020EncoderLayerWeights>::value,
              "encoder layer must expose W11");
static_assert(HasW12<hiko_s::ProteinMpnnV48020EncoderLayerWeights>::value,
              "encoder layer must expose W12");
static_assert(HasW13<hiko_s::ProteinMpnnV48020EncoderLayerWeights>::value,
              "encoder layer must expose W13");
static_assert(HasNorm3<hiko_s::ProteinMpnnV48020EncoderLayerWeights>::value,
              "encoder layer must expose norm3");
static_assert(!HasW11<hiko_s::ProteinMpnnV48020DecoderLayerWeights>::value,
              "decoder layer must not expose W11");
static_assert(!HasW12<hiko_s::ProteinMpnnV48020DecoderLayerWeights>::value,
              "decoder layer must not expose W12");
static_assert(!HasW13<hiko_s::ProteinMpnnV48020DecoderLayerWeights>::value,
              "decoder layer must not expose W13");
static_assert(!HasNorm3<hiko_s::ProteinMpnnV48020DecoderLayerWeights>::value,
              "decoder layer must not expose norm3");

namespace {

struct ExpectedTensor {
  std::string_view name;
  std::array<std::size_t, 2> shape;
  std::size_t rank;
};

constexpr ExpectedTensor expected_tensor(std::string_view name,
                                         std::size_t dim0) noexcept {
  return {name, {dim0, 0}, 1};
}

constexpr ExpectedTensor expected_tensor(std::string_view name,
                                         std::size_t dim0,
                                         std::size_t dim1) noexcept {
  return {name, {dim0, dim1}, 2};
}

constexpr bool string_equal(std::string_view lhs, std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index] != rhs[index]) {
      return false;
    }
  }
  return true;
}

constexpr std::size_t element_count(const ExpectedTensor& tensor) noexcept {
  return tensor.rank == 1 ? tensor.shape[0] : tensor.shape[0] * tensor.shape[1];
}

#define HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_ENCODER_LAYER(layer)                  \
  expected_tensor("encoder_layers." #layer ".norm1.weight", 128),              \
      expected_tensor("encoder_layers." #layer ".norm1.bias", 128),            \
      expected_tensor("encoder_layers." #layer ".norm2.weight", 128),          \
      expected_tensor("encoder_layers." #layer ".norm2.bias", 128),            \
      expected_tensor("encoder_layers." #layer ".norm3.weight", 128),          \
      expected_tensor("encoder_layers." #layer ".norm3.bias", 128),            \
      expected_tensor("encoder_layers." #layer ".W1.weight", 128, 384),        \
      expected_tensor("encoder_layers." #layer ".W1.bias", 128),               \
      expected_tensor("encoder_layers." #layer ".W2.weight", 128, 128),        \
      expected_tensor("encoder_layers." #layer ".W2.bias", 128),               \
      expected_tensor("encoder_layers." #layer ".W3.weight", 128, 128),        \
      expected_tensor("encoder_layers." #layer ".W3.bias", 128),               \
      expected_tensor("encoder_layers." #layer ".W11.weight", 128, 384),       \
      expected_tensor("encoder_layers." #layer ".W11.bias", 128),              \
      expected_tensor("encoder_layers." #layer ".W12.weight", 128, 128),       \
      expected_tensor("encoder_layers." #layer ".W12.bias", 128),              \
      expected_tensor("encoder_layers." #layer ".W13.weight", 128, 128),       \
      expected_tensor("encoder_layers." #layer ".W13.bias", 128),              \
      expected_tensor("encoder_layers." #layer ".dense.W_in.weight", 512, 128),\
      expected_tensor("encoder_layers." #layer ".dense.W_in.bias", 512),       \
      expected_tensor("encoder_layers." #layer ".dense.W_out.weight", 128, 512),\
      expected_tensor("encoder_layers." #layer ".dense.W_out.bias", 128)

#define HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_DECODER_LAYER(layer)                  \
  expected_tensor("decoder_layers." #layer ".norm1.weight", 128),              \
      expected_tensor("decoder_layers." #layer ".norm1.bias", 128),            \
      expected_tensor("decoder_layers." #layer ".norm2.weight", 128),          \
      expected_tensor("decoder_layers." #layer ".norm2.bias", 128),            \
      expected_tensor("decoder_layers." #layer ".W1.weight", 128, 512),        \
      expected_tensor("decoder_layers." #layer ".W1.bias", 128),               \
      expected_tensor("decoder_layers." #layer ".W2.weight", 128, 128),        \
      expected_tensor("decoder_layers." #layer ".W2.bias", 128),               \
      expected_tensor("decoder_layers." #layer ".W3.weight", 128, 128),        \
      expected_tensor("decoder_layers." #layer ".W3.bias", 128),               \
      expected_tensor("decoder_layers." #layer ".dense.W_in.weight", 512, 128),\
      expected_tensor("decoder_layers." #layer ".dense.W_in.bias", 512),       \
      expected_tensor("decoder_layers." #layer ".dense.W_out.weight", 128, 512),\
      expected_tensor("decoder_layers." #layer ".dense.W_out.bias", 128)

inline constexpr ExpectedTensor kExpectedSchema[] = {
    expected_tensor("features.embeddings.linear.weight", 16, 66),
    expected_tensor("features.embeddings.linear.bias", 16),
    expected_tensor("features.edge_embedding.weight", 128, 416),
    expected_tensor("features.norm_edges.weight", 128),
    expected_tensor("features.norm_edges.bias", 128),
    expected_tensor("W_e.weight", 128, 128),
    expected_tensor("W_e.bias", 128),
    expected_tensor("W_s.weight", 21, 128),
    HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_ENCODER_LAYER(0),
    HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_ENCODER_LAYER(1),
    HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_ENCODER_LAYER(2),
    HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_DECODER_LAYER(0),
    HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_DECODER_LAYER(1),
    HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_DECODER_LAYER(2),
    expected_tensor("W_out.weight", 21, 128),
    expected_tensor("W_out.bias", 21),
};

constexpr bool schema_matches_expected() noexcept {
  if (hiko_s::kProteinMpnnV48020SchemaTensorCount !=
      sizeof(kExpectedSchema) / sizeof(kExpectedSchema[0])) {
    return false;
  }
  for (std::size_t index = 0; index < hiko_s::kProteinMpnnV48020SchemaTensorCount;
       ++index) {
    const auto& actual = hiko_s::kProteinMpnnV48020TensorSchema[index];
    const auto& expected = kExpectedSchema[index];
    if (!string_equal(actual.name, expected.name) ||
        actual.rank != expected.rank ||
        actual.shape[0] != expected.shape[0] ||
        actual.shape[1] != expected.shape[1] ||
        actual.element_count != element_count(expected)) {
      return false;
    }
  }
  return true;
}

static_assert(schema_matches_expected(),
              "ProteinMPNN v_48_020 tensor schema must match checkpoint");

#undef HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_ENCODER_LAYER
#undef HIKOBOSHI_EXPECT_PROTEINMPNN_V48_020_DECODER_LAYER

}  // namespace

int main() {
  return 0;
}
