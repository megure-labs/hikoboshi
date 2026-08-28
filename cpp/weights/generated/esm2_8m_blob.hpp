#ifndef HIKOBOSHI_WEIGHTS_GENERATED_ESM2_8M_BLOB_HPP
#define HIKOBOSHI_WEIGHTS_GENERATED_ESM2_8M_BLOB_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hikoboshi::weights::generated::esm2_8m {

struct TensorBlobInfo {
  std::string_view name;
  std::size_t data_offset;
  std::size_t byte_length;
  const std::size_t* shape;
  const std::size_t* strides;
  std::size_t rank;
  std::string_view dtype;
  std::string_view checksum;
};

inline constexpr std::size_t kSafetensorsBlobLength = 29640016;
inline constexpr std::size_t kSafetensorsHeaderLength = 9288;
inline constexpr std::size_t kSafetensorsDataOffset = 9296;
inline constexpr std::size_t kSafetensorsDataLength = 29630720;
inline constexpr std::size_t kArchivedTensorCount = 99;
inline constexpr std::size_t kRuntimeTensorCount = 99;
inline constexpr std::string_view kSafetensorsBlobSha256{"7b1fea896ddbe6abb41e0f171cd1f9cce74868fbe44563afb87f3b176ccaca03"};
inline constexpr std::string_view kSafetensorsDataSha256{"315bdd0d2fe7382afa9f84ea9048d7c5deb87d310d05bd84e020f0f5619a6dca"};
inline constexpr std::string_view kSourceArtifactSha256{"afa959cb694cfbe7358b06f5c04eb9c67e8c36f89e7b2518cb6e3f8832d0d9ce"};
inline constexpr std::string_view kTensorSchema{"safetensors:hikoboshi-esm2-8m-v1;runtime_tensors=99"};
inline constexpr std::string_view kIgnoredHistoricalTensorNames[0] = {};

inline constexpr std::size_t kTensor0Shape[] = {29, 320};
inline constexpr std::size_t kTensor0Strides[] = {320, 1};
inline constexpr std::size_t kTensor1Shape[] = {320};
inline constexpr std::size_t kTensor1Strides[] = {1};
inline constexpr std::size_t kTensor2Shape[] = {320};
inline constexpr std::size_t kTensor2Strides[] = {1};
inline constexpr std::size_t kTensor3Shape[] = {320};
inline constexpr std::size_t kTensor3Strides[] = {1};
inline constexpr std::size_t kTensor4Shape[] = {320, 320};
inline constexpr std::size_t kTensor4Strides[] = {320, 1};
inline constexpr std::size_t kTensor5Shape[] = {320};
inline constexpr std::size_t kTensor5Strides[] = {1};
inline constexpr std::size_t kTensor6Shape[] = {320, 320};
inline constexpr std::size_t kTensor6Strides[] = {320, 1};
inline constexpr std::size_t kTensor7Shape[] = {320};
inline constexpr std::size_t kTensor7Strides[] = {1};
inline constexpr std::size_t kTensor8Shape[] = {320};
inline constexpr std::size_t kTensor8Strides[] = {1};
inline constexpr std::size_t kTensor9Shape[] = {320};
inline constexpr std::size_t kTensor9Strides[] = {1};
inline constexpr std::size_t kTensor10Shape[] = {320, 320};
inline constexpr std::size_t kTensor10Strides[] = {320, 1};
inline constexpr std::size_t kTensor11Shape[] = {320};
inline constexpr std::size_t kTensor11Strides[] = {1};
inline constexpr std::size_t kTensor12Shape[] = {320, 320};
inline constexpr std::size_t kTensor12Strides[] = {320, 1};
inline constexpr std::size_t kTensor13Shape[] = {1280};
inline constexpr std::size_t kTensor13Strides[] = {1};
inline constexpr std::size_t kTensor14Shape[] = {1280, 320};
inline constexpr std::size_t kTensor14Strides[] = {320, 1};
inline constexpr std::size_t kTensor15Shape[] = {320};
inline constexpr std::size_t kTensor15Strides[] = {1};
inline constexpr std::size_t kTensor16Shape[] = {320, 1280};
inline constexpr std::size_t kTensor16Strides[] = {1280, 1};
inline constexpr std::size_t kTensor17Shape[] = {320};
inline constexpr std::size_t kTensor17Strides[] = {1};
inline constexpr std::size_t kTensor18Shape[] = {320};
inline constexpr std::size_t kTensor18Strides[] = {1};
inline constexpr std::size_t kTensor19Shape[] = {320};
inline constexpr std::size_t kTensor19Strides[] = {1};
inline constexpr std::size_t kTensor20Shape[] = {320, 320};
inline constexpr std::size_t kTensor20Strides[] = {320, 1};
inline constexpr std::size_t kTensor21Shape[] = {320};
inline constexpr std::size_t kTensor21Strides[] = {1};
inline constexpr std::size_t kTensor22Shape[] = {320, 320};
inline constexpr std::size_t kTensor22Strides[] = {320, 1};
inline constexpr std::size_t kTensor23Shape[] = {320};
inline constexpr std::size_t kTensor23Strides[] = {1};
inline constexpr std::size_t kTensor24Shape[] = {320};
inline constexpr std::size_t kTensor24Strides[] = {1};
inline constexpr std::size_t kTensor25Shape[] = {320};
inline constexpr std::size_t kTensor25Strides[] = {1};
inline constexpr std::size_t kTensor26Shape[] = {320, 320};
inline constexpr std::size_t kTensor26Strides[] = {320, 1};
inline constexpr std::size_t kTensor27Shape[] = {320};
inline constexpr std::size_t kTensor27Strides[] = {1};
inline constexpr std::size_t kTensor28Shape[] = {320, 320};
inline constexpr std::size_t kTensor28Strides[] = {320, 1};
inline constexpr std::size_t kTensor29Shape[] = {1280};
inline constexpr std::size_t kTensor29Strides[] = {1};
inline constexpr std::size_t kTensor30Shape[] = {1280, 320};
inline constexpr std::size_t kTensor30Strides[] = {320, 1};
inline constexpr std::size_t kTensor31Shape[] = {320};
inline constexpr std::size_t kTensor31Strides[] = {1};
inline constexpr std::size_t kTensor32Shape[] = {320, 1280};
inline constexpr std::size_t kTensor32Strides[] = {1280, 1};
inline constexpr std::size_t kTensor33Shape[] = {320};
inline constexpr std::size_t kTensor33Strides[] = {1};
inline constexpr std::size_t kTensor34Shape[] = {320};
inline constexpr std::size_t kTensor34Strides[] = {1};
inline constexpr std::size_t kTensor35Shape[] = {320};
inline constexpr std::size_t kTensor35Strides[] = {1};
inline constexpr std::size_t kTensor36Shape[] = {320, 320};
inline constexpr std::size_t kTensor36Strides[] = {320, 1};
inline constexpr std::size_t kTensor37Shape[] = {320};
inline constexpr std::size_t kTensor37Strides[] = {1};
inline constexpr std::size_t kTensor38Shape[] = {320, 320};
inline constexpr std::size_t kTensor38Strides[] = {320, 1};
inline constexpr std::size_t kTensor39Shape[] = {320};
inline constexpr std::size_t kTensor39Strides[] = {1};
inline constexpr std::size_t kTensor40Shape[] = {320};
inline constexpr std::size_t kTensor40Strides[] = {1};
inline constexpr std::size_t kTensor41Shape[] = {320};
inline constexpr std::size_t kTensor41Strides[] = {1};
inline constexpr std::size_t kTensor42Shape[] = {320, 320};
inline constexpr std::size_t kTensor42Strides[] = {320, 1};
inline constexpr std::size_t kTensor43Shape[] = {320};
inline constexpr std::size_t kTensor43Strides[] = {1};
inline constexpr std::size_t kTensor44Shape[] = {320, 320};
inline constexpr std::size_t kTensor44Strides[] = {320, 1};
inline constexpr std::size_t kTensor45Shape[] = {1280};
inline constexpr std::size_t kTensor45Strides[] = {1};
inline constexpr std::size_t kTensor46Shape[] = {1280, 320};
inline constexpr std::size_t kTensor46Strides[] = {320, 1};
inline constexpr std::size_t kTensor47Shape[] = {320};
inline constexpr std::size_t kTensor47Strides[] = {1};
inline constexpr std::size_t kTensor48Shape[] = {320, 1280};
inline constexpr std::size_t kTensor48Strides[] = {1280, 1};
inline constexpr std::size_t kTensor49Shape[] = {320};
inline constexpr std::size_t kTensor49Strides[] = {1};
inline constexpr std::size_t kTensor50Shape[] = {320};
inline constexpr std::size_t kTensor50Strides[] = {1};
inline constexpr std::size_t kTensor51Shape[] = {320};
inline constexpr std::size_t kTensor51Strides[] = {1};
inline constexpr std::size_t kTensor52Shape[] = {320, 320};
inline constexpr std::size_t kTensor52Strides[] = {320, 1};
inline constexpr std::size_t kTensor53Shape[] = {320};
inline constexpr std::size_t kTensor53Strides[] = {1};
inline constexpr std::size_t kTensor54Shape[] = {320, 320};
inline constexpr std::size_t kTensor54Strides[] = {320, 1};
inline constexpr std::size_t kTensor55Shape[] = {320};
inline constexpr std::size_t kTensor55Strides[] = {1};
inline constexpr std::size_t kTensor56Shape[] = {320};
inline constexpr std::size_t kTensor56Strides[] = {1};
inline constexpr std::size_t kTensor57Shape[] = {320};
inline constexpr std::size_t kTensor57Strides[] = {1};
inline constexpr std::size_t kTensor58Shape[] = {320, 320};
inline constexpr std::size_t kTensor58Strides[] = {320, 1};
inline constexpr std::size_t kTensor59Shape[] = {320};
inline constexpr std::size_t kTensor59Strides[] = {1};
inline constexpr std::size_t kTensor60Shape[] = {320, 320};
inline constexpr std::size_t kTensor60Strides[] = {320, 1};
inline constexpr std::size_t kTensor61Shape[] = {1280};
inline constexpr std::size_t kTensor61Strides[] = {1};
inline constexpr std::size_t kTensor62Shape[] = {1280, 320};
inline constexpr std::size_t kTensor62Strides[] = {320, 1};
inline constexpr std::size_t kTensor63Shape[] = {320};
inline constexpr std::size_t kTensor63Strides[] = {1};
inline constexpr std::size_t kTensor64Shape[] = {320, 1280};
inline constexpr std::size_t kTensor64Strides[] = {1280, 1};
inline constexpr std::size_t kTensor65Shape[] = {320};
inline constexpr std::size_t kTensor65Strides[] = {1};
inline constexpr std::size_t kTensor66Shape[] = {320};
inline constexpr std::size_t kTensor66Strides[] = {1};
inline constexpr std::size_t kTensor67Shape[] = {320};
inline constexpr std::size_t kTensor67Strides[] = {1};
inline constexpr std::size_t kTensor68Shape[] = {320, 320};
inline constexpr std::size_t kTensor68Strides[] = {320, 1};
inline constexpr std::size_t kTensor69Shape[] = {320};
inline constexpr std::size_t kTensor69Strides[] = {1};
inline constexpr std::size_t kTensor70Shape[] = {320, 320};
inline constexpr std::size_t kTensor70Strides[] = {320, 1};
inline constexpr std::size_t kTensor71Shape[] = {320};
inline constexpr std::size_t kTensor71Strides[] = {1};
inline constexpr std::size_t kTensor72Shape[] = {320};
inline constexpr std::size_t kTensor72Strides[] = {1};
inline constexpr std::size_t kTensor73Shape[] = {320};
inline constexpr std::size_t kTensor73Strides[] = {1};
inline constexpr std::size_t kTensor74Shape[] = {320, 320};
inline constexpr std::size_t kTensor74Strides[] = {320, 1};
inline constexpr std::size_t kTensor75Shape[] = {320};
inline constexpr std::size_t kTensor75Strides[] = {1};
inline constexpr std::size_t kTensor76Shape[] = {320, 320};
inline constexpr std::size_t kTensor76Strides[] = {320, 1};
inline constexpr std::size_t kTensor77Shape[] = {1280};
inline constexpr std::size_t kTensor77Strides[] = {1};
inline constexpr std::size_t kTensor78Shape[] = {1280, 320};
inline constexpr std::size_t kTensor78Strides[] = {320, 1};
inline constexpr std::size_t kTensor79Shape[] = {320};
inline constexpr std::size_t kTensor79Strides[] = {1};
inline constexpr std::size_t kTensor80Shape[] = {320, 1280};
inline constexpr std::size_t kTensor80Strides[] = {1280, 1};
inline constexpr std::size_t kTensor81Shape[] = {320};
inline constexpr std::size_t kTensor81Strides[] = {1};
inline constexpr std::size_t kTensor82Shape[] = {320};
inline constexpr std::size_t kTensor82Strides[] = {1};
inline constexpr std::size_t kTensor83Shape[] = {320};
inline constexpr std::size_t kTensor83Strides[] = {1};
inline constexpr std::size_t kTensor84Shape[] = {320, 320};
inline constexpr std::size_t kTensor84Strides[] = {320, 1};
inline constexpr std::size_t kTensor85Shape[] = {320};
inline constexpr std::size_t kTensor85Strides[] = {1};
inline constexpr std::size_t kTensor86Shape[] = {320, 320};
inline constexpr std::size_t kTensor86Strides[] = {320, 1};
inline constexpr std::size_t kTensor87Shape[] = {320};
inline constexpr std::size_t kTensor87Strides[] = {1};
inline constexpr std::size_t kTensor88Shape[] = {320};
inline constexpr std::size_t kTensor88Strides[] = {1};
inline constexpr std::size_t kTensor89Shape[] = {320};
inline constexpr std::size_t kTensor89Strides[] = {1};
inline constexpr std::size_t kTensor90Shape[] = {320, 320};
inline constexpr std::size_t kTensor90Strides[] = {320, 1};
inline constexpr std::size_t kTensor91Shape[] = {320};
inline constexpr std::size_t kTensor91Strides[] = {1};
inline constexpr std::size_t kTensor92Shape[] = {320, 320};
inline constexpr std::size_t kTensor92Strides[] = {320, 1};
inline constexpr std::size_t kTensor93Shape[] = {1280};
inline constexpr std::size_t kTensor93Strides[] = {1};
inline constexpr std::size_t kTensor94Shape[] = {1280, 320};
inline constexpr std::size_t kTensor94Strides[] = {320, 1};
inline constexpr std::size_t kTensor95Shape[] = {320};
inline constexpr std::size_t kTensor95Strides[] = {1};
inline constexpr std::size_t kTensor96Shape[] = {320, 1280};
inline constexpr std::size_t kTensor96Strides[] = {1280, 1};
inline constexpr std::size_t kTensor97Shape[] = {320};
inline constexpr std::size_t kTensor97Strides[] = {1};
inline constexpr std::size_t kTensor98Shape[] = {320};
inline constexpr std::size_t kTensor98Strides[] = {1};

inline constexpr TensorBlobInfo kRuntimeTensors[] = {
    {"embedding_table", 0, 37120, kTensor0Shape, kTensor0Strides, 2, "float32", "686fc7434fb3c5f5459013dc0521c3bca7c451fd1f6f0bb68dad6485100c77e3"},
    {"final_norm.bias", 37120, 1280, kTensor1Shape, kTensor1Strides, 1, "float32", "e1ac2397485287673738c359e7fefff872f8920acf9a0d3c2195746f97f5c5e6"},
    {"final_norm.weight", 38400, 1280, kTensor2Shape, kTensor2Strides, 1, "float32", "58a3adfd87a6bb5f9326e043b63528538c6f9ec8f7ce808483669120316c6bee"},
    {"layers.0.attn.k_proj.bias", 39680, 1280, kTensor3Shape, kTensor3Strides, 1, "float32", "c2814dc8345df07f6d09df64ff8218b318c7e1f1334ccb163496301dc73f22bf"},
    {"layers.0.attn.k_proj.weight", 40960, 409600, kTensor4Shape, kTensor4Strides, 2, "float32", "e74906c59cc8606edd8e8f50739281b62094ab31625e03db2cae8958b481fd89"},
    {"layers.0.attn.out_proj.bias", 450560, 1280, kTensor5Shape, kTensor5Strides, 1, "float32", "323db1436b6532e2300c26a2f7a0e5fa86cf2be6157e53b22fddde2f7c419c0b"},
    {"layers.0.attn.out_proj.weight", 451840, 409600, kTensor6Shape, kTensor6Strides, 2, "float32", "5dccd001e775f814ae496c20f65c3d6442e04dd8ffb44e9768b6adbbe325ec81"},
    {"layers.0.attn.pre_norm.bias", 861440, 1280, kTensor7Shape, kTensor7Strides, 1, "float32", "3887e9b333fd783c974844b2ce4c58fa061e439d7b79d9aa0f61bc45a84c1aa6"},
    {"layers.0.attn.pre_norm.weight", 862720, 1280, kTensor8Shape, kTensor8Strides, 1, "float32", "6df22b063a4bfd40f1c9ceef397ac68ed822c695fdbe36434a51fe62d97bd122"},
    {"layers.0.attn.q_proj.bias", 864000, 1280, kTensor9Shape, kTensor9Strides, 1, "float32", "03329d426c01be4172a811aaf401619f8614d2ef578eba449829a00db8767282"},
    {"layers.0.attn.q_proj.weight", 865280, 409600, kTensor10Shape, kTensor10Strides, 2, "float32", "b54e710f1e1838141eb81835f5bec0bd60c90af8a30fe57eb9feb261e483cf9c"},
    {"layers.0.attn.v_proj.bias", 1274880, 1280, kTensor11Shape, kTensor11Strides, 1, "float32", "8d1f4e3c636d7d77278367c582107a468fe119bfd5d8860abfdc9459a77ebc6f"},
    {"layers.0.attn.v_proj.weight", 1276160, 409600, kTensor12Shape, kTensor12Strides, 2, "float32", "abd8d82a9fac20b3059016366e3e8b2641c1276dd05fca5aee7a4f067a1449fe"},
    {"layers.0.ffn.in.bias", 1685760, 5120, kTensor13Shape, kTensor13Strides, 1, "float32", "f2e9f6d47818fd12ca11da4cd52e8251946ffc09c6e2b4b8fb9bdd18f59433f6"},
    {"layers.0.ffn.in.weight", 1690880, 1638400, kTensor14Shape, kTensor14Strides, 2, "float32", "b2b29e814f40a8dd1748f3c001a3780338a9fe5ccb1510091e16323fb4b8d58a"},
    {"layers.0.ffn.out.bias", 3329280, 1280, kTensor15Shape, kTensor15Strides, 1, "float32", "630042b77aa71602e326cc8a9fe4b3e604335503d731e0a0a669b5b01d4077e6"},
    {"layers.0.ffn.out.weight", 3330560, 1638400, kTensor16Shape, kTensor16Strides, 2, "float32", "3f2903213b207ee65b81583fc0901b0d2bb7777075b753fb95b277ac965f2e50"},
    {"layers.0.ffn.pre_norm.bias", 4968960, 1280, kTensor17Shape, kTensor17Strides, 1, "float32", "a47943275a98256839fb2318fc020d5108829c7d5a31b1c6e6957fd562205157"},
    {"layers.0.ffn.pre_norm.weight", 4970240, 1280, kTensor18Shape, kTensor18Strides, 1, "float32", "d557421433a72ed9b40bb7affdcc52d706747a61861cd0b77a2bc187ee798c43"},
    {"layers.1.attn.k_proj.bias", 4971520, 1280, kTensor19Shape, kTensor19Strides, 1, "float32", "53740485b38b514d466a34f533f2500f14ac463db14e9268a8305eede9e459ef"},
    {"layers.1.attn.k_proj.weight", 4972800, 409600, kTensor20Shape, kTensor20Strides, 2, "float32", "be6da96166ac291232413dac800e84315846700db4b36a93548a171f1ebccc2c"},
    {"layers.1.attn.out_proj.bias", 5382400, 1280, kTensor21Shape, kTensor21Strides, 1, "float32", "41f2263e2368d5a6202532872410894b9c6e5600bff27299ee759593a917c829"},
    {"layers.1.attn.out_proj.weight", 5383680, 409600, kTensor22Shape, kTensor22Strides, 2, "float32", "02eb6a2fcbabe126b330696a7f47e2e8b1fa0085ccbd0d1631dcf942c27a3752"},
    {"layers.1.attn.pre_norm.bias", 5793280, 1280, kTensor23Shape, kTensor23Strides, 1, "float32", "6624d1a5310ddf6a2cdf0a5db01a25b256784a450f429e98d70075dae334a429"},
    {"layers.1.attn.pre_norm.weight", 5794560, 1280, kTensor24Shape, kTensor24Strides, 1, "float32", "909bb9e9cb7b4882dd9ff40e42f600211f76518e1090ca89c416cf14b9673b34"},
    {"layers.1.attn.q_proj.bias", 5795840, 1280, kTensor25Shape, kTensor25Strides, 1, "float32", "3c53ed604c8e6789cb269eb35c7d4582959545906b1cf5172a747388aeda6a04"},
    {"layers.1.attn.q_proj.weight", 5797120, 409600, kTensor26Shape, kTensor26Strides, 2, "float32", "d1761f3f07f7a3db357601aff3c427bee5d524e15786fb581ca0d80431118cba"},
    {"layers.1.attn.v_proj.bias", 6206720, 1280, kTensor27Shape, kTensor27Strides, 1, "float32", "c0db7ca4ddd24c5ea0a1fb0bc511f7610b50965721ef3728853af2a352ab3554"},
    {"layers.1.attn.v_proj.weight", 6208000, 409600, kTensor28Shape, kTensor28Strides, 2, "float32", "287f36032775d09c4339e034ccbbf9f8b709e24cec6f2e6de6f9763cdb066c89"},
    {"layers.1.ffn.in.bias", 6617600, 5120, kTensor29Shape, kTensor29Strides, 1, "float32", "3f9e140cc3a32eb2772151c0ac0342abbd8d60dcc87dacdc00c2d201fee1437b"},
    {"layers.1.ffn.in.weight", 6622720, 1638400, kTensor30Shape, kTensor30Strides, 2, "float32", "d604535c873a5bc9a95c8bc4fa94c61d5195e5d8db5bacf40dc2a5ce5c1b583c"},
    {"layers.1.ffn.out.bias", 8261120, 1280, kTensor31Shape, kTensor31Strides, 1, "float32", "0e9bf1f424980c5ae6b68b1414bbee63929a0e2c792ac2efb22fcfdb97462695"},
    {"layers.1.ffn.out.weight", 8262400, 1638400, kTensor32Shape, kTensor32Strides, 2, "float32", "5b71edc161ef0a28ad3045e5244df79b0a5c99d35e526215a5b98b31da28d62b"},
    {"layers.1.ffn.pre_norm.bias", 9900800, 1280, kTensor33Shape, kTensor33Strides, 1, "float32", "5afd271dfd36b60a5be9d31ea67208e989ea9dc1eff6fc05f9e6779567fa4e33"},
    {"layers.1.ffn.pre_norm.weight", 9902080, 1280, kTensor34Shape, kTensor34Strides, 1, "float32", "b28388486658215b1d0a35532ece9f68bb85809566ce82411169b5a1d0dbc43a"},
    {"layers.2.attn.k_proj.bias", 9903360, 1280, kTensor35Shape, kTensor35Strides, 1, "float32", "350905843962d6a963802d91a832c1c5edde5603c3e5937cfb6a31fc60ac8b5e"},
    {"layers.2.attn.k_proj.weight", 9904640, 409600, kTensor36Shape, kTensor36Strides, 2, "float32", "94df685fdae2390ab9c41ad18d16c325c9adb4cd404483da17d869845b248594"},
    {"layers.2.attn.out_proj.bias", 10314240, 1280, kTensor37Shape, kTensor37Strides, 1, "float32", "3e068925f85323f31d248bfe82c23d01fb0e288d17eb3e06e2b0dab688ea77c5"},
    {"layers.2.attn.out_proj.weight", 10315520, 409600, kTensor38Shape, kTensor38Strides, 2, "float32", "1814b03d35578c67b9cfbfd7095761ec741a14c46146e4014ab56514f13dd336"},
    {"layers.2.attn.pre_norm.bias", 10725120, 1280, kTensor39Shape, kTensor39Strides, 1, "float32", "ca94ef5ceb5e4233539fa0e9e634b67ee078754753d4a03901f6ab656288c312"},
    {"layers.2.attn.pre_norm.weight", 10726400, 1280, kTensor40Shape, kTensor40Strides, 1, "float32", "11e3d0c4be9b9752f265581f908e6213f2619316368f827dfca321e697f6e910"},
    {"layers.2.attn.q_proj.bias", 10727680, 1280, kTensor41Shape, kTensor41Strides, 1, "float32", "6a01279444282b120f6a7f361bc347de586d6ad822e68dddd9d89d8a503e302f"},
    {"layers.2.attn.q_proj.weight", 10728960, 409600, kTensor42Shape, kTensor42Strides, 2, "float32", "f9ae9d79c41c4b5e170ee7a1d25b5b2074f039a221e9a98d6639b9440350124a"},
    {"layers.2.attn.v_proj.bias", 11138560, 1280, kTensor43Shape, kTensor43Strides, 1, "float32", "64840ed7c113581e1f705cb9eb830d78fd3f0f4a1da0c52caebd0c5845d20f94"},
    {"layers.2.attn.v_proj.weight", 11139840, 409600, kTensor44Shape, kTensor44Strides, 2, "float32", "9572970e5604860ce3ccd99a149f88f27af3a511164da442bd6b151233729a52"},
    {"layers.2.ffn.in.bias", 11549440, 5120, kTensor45Shape, kTensor45Strides, 1, "float32", "0b140cc3c65de1d2371f765a2be5fc9e6614895e66bdd96761a1c69bdadd6453"},
    {"layers.2.ffn.in.weight", 11554560, 1638400, kTensor46Shape, kTensor46Strides, 2, "float32", "c719ed8dd4f962c2b29bd85e167102e16cbd42566dbddcda75a2b42563a2a3f2"},
    {"layers.2.ffn.out.bias", 13192960, 1280, kTensor47Shape, kTensor47Strides, 1, "float32", "ed0ed59694026f348d38a47d8c2a3b5c6ada302461ac63232edbc19a0f2662d0"},
    {"layers.2.ffn.out.weight", 13194240, 1638400, kTensor48Shape, kTensor48Strides, 2, "float32", "50871c92b93752d360ba09d91572189f19f768eaaa4b9c6403b76895593da0c3"},
    {"layers.2.ffn.pre_norm.bias", 14832640, 1280, kTensor49Shape, kTensor49Strides, 1, "float32", "226783aff67bc95f67c1a5a23d604ee20d017dd819e9fdf936859bcb25f3c0c6"},
    {"layers.2.ffn.pre_norm.weight", 14833920, 1280, kTensor50Shape, kTensor50Strides, 1, "float32", "876db6cdfd16f64c05850459795cf7c8ab222d928056298afc741eb7708742b1"},
    {"layers.3.attn.k_proj.bias", 14835200, 1280, kTensor51Shape, kTensor51Strides, 1, "float32", "bbe7ffb787e78f689d1a870275d806fd0226e4cfa3a5ea92275920eddf2473de"},
    {"layers.3.attn.k_proj.weight", 14836480, 409600, kTensor52Shape, kTensor52Strides, 2, "float32", "45e7fbf98ca13fd1f3e3216a7efcb73cc61a5cf9fc3d61f5b808114b046c26c8"},
    {"layers.3.attn.out_proj.bias", 15246080, 1280, kTensor53Shape, kTensor53Strides, 1, "float32", "903f1da54909134ba873f869594584a3beb9cd49c96ef5fab7fd32c37de7c09f"},
    {"layers.3.attn.out_proj.weight", 15247360, 409600, kTensor54Shape, kTensor54Strides, 2, "float32", "7eee298e489034b4d434e6296bcc76e324ffbaf01d338d9817991292c9aae456"},
    {"layers.3.attn.pre_norm.bias", 15656960, 1280, kTensor55Shape, kTensor55Strides, 1, "float32", "0dec1dccde16f13d8480d6be6b0db5497606de1eb6684a2ee67d55ab1b663f6d"},
    {"layers.3.attn.pre_norm.weight", 15658240, 1280, kTensor56Shape, kTensor56Strides, 1, "float32", "94f90f66ec49715a73f1c6104f8aa84a92ebd1ca3a7170667b43ea45a9949f35"},
    {"layers.3.attn.q_proj.bias", 15659520, 1280, kTensor57Shape, kTensor57Strides, 1, "float32", "427ec156df5a218303f39f72cc38e06b32eddffdd4912d3cb756c0ee8210c852"},
    {"layers.3.attn.q_proj.weight", 15660800, 409600, kTensor58Shape, kTensor58Strides, 2, "float32", "3d37d22a3ce47ca58f823f8c3f6f6d5200cbc8fa8aee8404f6d5fb8ee004c3d7"},
    {"layers.3.attn.v_proj.bias", 16070400, 1280, kTensor59Shape, kTensor59Strides, 1, "float32", "33563234331803526d387ecba331f30b1e13706d4b5e955c1964d8bbaaf2da0b"},
    {"layers.3.attn.v_proj.weight", 16071680, 409600, kTensor60Shape, kTensor60Strides, 2, "float32", "1371a18da2c24dcea5e3462e5ef44cf3470f2c82a66845a0041adc31feacf2fe"},
    {"layers.3.ffn.in.bias", 16481280, 5120, kTensor61Shape, kTensor61Strides, 1, "float32", "f30303facac72b53e44a01110df5b2319d570e2e5b987b6cc3f751b3483c69b5"},
    {"layers.3.ffn.in.weight", 16486400, 1638400, kTensor62Shape, kTensor62Strides, 2, "float32", "3be013f205d0bb64bd6faea4530b890201a2110f62621ca32a0fac51127e979a"},
    {"layers.3.ffn.out.bias", 18124800, 1280, kTensor63Shape, kTensor63Strides, 1, "float32", "ce9afab4aaf1707c9cd845576b39f9f2c9af78aa1ee7ec2d767a076eb99db23d"},
    {"layers.3.ffn.out.weight", 18126080, 1638400, kTensor64Shape, kTensor64Strides, 2, "float32", "7431c1d354f4edcf560b7afc28033a5af1724c382c4598a2999e6c66f8e0b809"},
    {"layers.3.ffn.pre_norm.bias", 19764480, 1280, kTensor65Shape, kTensor65Strides, 1, "float32", "7b9cb8b343c5eabf6a8805783621b465991b9d55d9ea9dc732e6865d3c669d9a"},
    {"layers.3.ffn.pre_norm.weight", 19765760, 1280, kTensor66Shape, kTensor66Strides, 1, "float32", "32b0c8390303184d02779a63637dae89685f5114a27abc10532fba9fba875047"},
    {"layers.4.attn.k_proj.bias", 19767040, 1280, kTensor67Shape, kTensor67Strides, 1, "float32", "915f5012c264c9fc534d6f6628bada49a49eebc5a87a4cc6c22b6d1080542ddb"},
    {"layers.4.attn.k_proj.weight", 19768320, 409600, kTensor68Shape, kTensor68Strides, 2, "float32", "a45dcd27e7b6fa5a0a019037e376c77f5b6ba4a20db6e9f837cd74ff20dfa370"},
    {"layers.4.attn.out_proj.bias", 20177920, 1280, kTensor69Shape, kTensor69Strides, 1, "float32", "ee5c64dc315df21a548f65232620d155847a584d84f0d99c4a497d9819f50a5b"},
    {"layers.4.attn.out_proj.weight", 20179200, 409600, kTensor70Shape, kTensor70Strides, 2, "float32", "5a9bd16eda2addec40f6dd73cb4f185e65abf02c3d99c07fb17f4335363d27dc"},
    {"layers.4.attn.pre_norm.bias", 20588800, 1280, kTensor71Shape, kTensor71Strides, 1, "float32", "487180156b3e1dc3fbc279102e845aba5f7522ca22157e2b5fd050890af82e71"},
    {"layers.4.attn.pre_norm.weight", 20590080, 1280, kTensor72Shape, kTensor72Strides, 1, "float32", "f65a14f503b655144d54aa8ae6741b16e8fec4c938ecc4dcb3998fa8e20dbd82"},
    {"layers.4.attn.q_proj.bias", 20591360, 1280, kTensor73Shape, kTensor73Strides, 1, "float32", "2707e8e503a246445f887c3cb64e916ae702d4c87b7475a2eb1be3c7a7535538"},
    {"layers.4.attn.q_proj.weight", 20592640, 409600, kTensor74Shape, kTensor74Strides, 2, "float32", "6ca9929658772c4c3351632e3c78b3913a558405de15f8858ecfd657fb7b33c2"},
    {"layers.4.attn.v_proj.bias", 21002240, 1280, kTensor75Shape, kTensor75Strides, 1, "float32", "b3527ad4cd3004bffc67240b63208d5df1bfda639413f7a0dee991a1877c8e6e"},
    {"layers.4.attn.v_proj.weight", 21003520, 409600, kTensor76Shape, kTensor76Strides, 2, "float32", "420f04c2fc3da0d205b35df94abe10e1569f63311eb46cec952526d7ee174846"},
    {"layers.4.ffn.in.bias", 21413120, 5120, kTensor77Shape, kTensor77Strides, 1, "float32", "ef995bad8144d945f439070ccc0b24251ace8d8d728e1981e26fc5b54aa791c4"},
    {"layers.4.ffn.in.weight", 21418240, 1638400, kTensor78Shape, kTensor78Strides, 2, "float32", "cd2f5236e8ba9710544b693ab03fdfb69ee45819a5011cb1f6f0267e786e0aae"},
    {"layers.4.ffn.out.bias", 23056640, 1280, kTensor79Shape, kTensor79Strides, 1, "float32", "7f7d91a61f7a47fd020f1939f8dd002e80480e451bd21833b5abe446634671be"},
    {"layers.4.ffn.out.weight", 23057920, 1638400, kTensor80Shape, kTensor80Strides, 2, "float32", "c07c2fe6c1588d09a370c1bacaf860fe53293831cefbdd55ae5c9ddd1f2db9d0"},
    {"layers.4.ffn.pre_norm.bias", 24696320, 1280, kTensor81Shape, kTensor81Strides, 1, "float32", "7813d8b7125ae68c78d24a1c6d00ecb39e7657272db7495d3f9e6660676251b1"},
    {"layers.4.ffn.pre_norm.weight", 24697600, 1280, kTensor82Shape, kTensor82Strides, 1, "float32", "dbaa6ca49f831ec64848663aa1dd3f57a27fd76dc4f79bc9bb42f85b36bf6cdb"},
    {"layers.5.attn.k_proj.bias", 24698880, 1280, kTensor83Shape, kTensor83Strides, 1, "float32", "ef497f68106189b1346b652e44d810866e2d41caf0259c6713e1132fd0275cb8"},
    {"layers.5.attn.k_proj.weight", 24700160, 409600, kTensor84Shape, kTensor84Strides, 2, "float32", "0c2fe07a8079ed6404b2083856d6ff00eb7f9b0c63ebfd96729de3eaf3b98750"},
    {"layers.5.attn.out_proj.bias", 25109760, 1280, kTensor85Shape, kTensor85Strides, 1, "float32", "cfcf8f93c3f93f5492f9bfaca90a4354e885232f7b2732edbff4f0f7c130e70a"},
    {"layers.5.attn.out_proj.weight", 25111040, 409600, kTensor86Shape, kTensor86Strides, 2, "float32", "706a922f0e91c2159a66544f6eb7a5c8d159bdd0dd1a4b66a75293d13f4c38e3"},
    {"layers.5.attn.pre_norm.bias", 25520640, 1280, kTensor87Shape, kTensor87Strides, 1, "float32", "a943dd9236a5348dd91b880f9cdc8f4712839c3be71826fd142ca319f1998e6f"},
    {"layers.5.attn.pre_norm.weight", 25521920, 1280, kTensor88Shape, kTensor88Strides, 1, "float32", "a81649aea8942c242980d2249f4f86ac8c111852a040280e9aa0a452c31b9d07"},
    {"layers.5.attn.q_proj.bias", 25523200, 1280, kTensor89Shape, kTensor89Strides, 1, "float32", "8965ed97c305d5152e9dee3e476c07b7b5d34e2f26d04cc8d5ecce6b49aa8c1e"},
    {"layers.5.attn.q_proj.weight", 25524480, 409600, kTensor90Shape, kTensor90Strides, 2, "float32", "b7c9d02d9b2e4129054c1048fa9edfb84ba2eeca93c0ca46546bbf95edc9ab4a"},
    {"layers.5.attn.v_proj.bias", 25934080, 1280, kTensor91Shape, kTensor91Strides, 1, "float32", "ac26e4b95f77ae32391ec6d2e25b899d51e2cf1be1dcd7f786a86d2b844a5454"},
    {"layers.5.attn.v_proj.weight", 25935360, 409600, kTensor92Shape, kTensor92Strides, 2, "float32", "72948aedd3d22a8b8efd8ecf52f25a4ebf99defaccf450755d49c68916316b9e"},
    {"layers.5.ffn.in.bias", 26344960, 5120, kTensor93Shape, kTensor93Strides, 1, "float32", "25a275c2b22dc78ddc02160dc9508166bbee2c7d727ec4d86fdb944edcb2ed1a"},
    {"layers.5.ffn.in.weight", 26350080, 1638400, kTensor94Shape, kTensor94Strides, 2, "float32", "a2536457b49598cb412b2364584d04b34df2f03da19f145a6dda8cde47696d6e"},
    {"layers.5.ffn.out.bias", 27988480, 1280, kTensor95Shape, kTensor95Strides, 1, "float32", "e30f02f2ab12a6c72b8ebd163c987cd883ace5a5fca23da340e458ea20e646da"},
    {"layers.5.ffn.out.weight", 27989760, 1638400, kTensor96Shape, kTensor96Strides, 2, "float32", "d0a101eda1bbc8045930903f6d6ad85810d88e44a2a7f81513e08c01c3b380d5"},
    {"layers.5.ffn.pre_norm.bias", 29628160, 1280, kTensor97Shape, kTensor97Strides, 1, "float32", "825ef4d4e2aed769d833229fd9b07f73827c0ece0087918cfdddb0510c12313f"},
    {"layers.5.ffn.pre_norm.weight", 29629440, 1280, kTensor98Shape, kTensor98Strides, 1, "float32", "171ac17ba42207c3b8f25881f0598f8a5c6251e026740f2ab4abed74ef03f69b"},
};

alignas(64) extern const std::uint8_t kSafetensorsBlob[kSafetensorsBlobLength];

static_assert(sizeof(kSafetensorsBlob) == kSafetensorsBlobLength,
              "Hikoboshi-esm2_8m blob length mismatch");
static_assert(kSafetensorsDataOffset + kSafetensorsDataLength ==
                  kSafetensorsBlobLength,
              "Hikoboshi-esm2_8m safetensors layout mismatch");

}  // namespace hikoboshi::weights::generated::esm2_8m

#endif  // HIKOBOSHI_WEIGHTS_GENERATED_ESM2_8M_BLOB_HPP
