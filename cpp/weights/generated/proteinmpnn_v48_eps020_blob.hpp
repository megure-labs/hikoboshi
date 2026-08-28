#ifndef HIKOBOSHI_WEIGHTS_GENERATED_PROTEINMPNN_V48_EPS020_BLOB_HPP
#define HIKOBOSHI_WEIGHTS_GENERATED_PROTEINMPNN_V48_EPS020_BLOB_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hikoboshi::weights::generated::proteinmpnn_v48_eps020 {

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

inline constexpr std::size_t kSafetensorsBlobLength = 6653068;
inline constexpr std::size_t kSafetensorsHeaderLength = 11120;
inline constexpr std::size_t kSafetensorsDataOffset = 11128;
inline constexpr std::size_t kSafetensorsDataLength = 6641940;
inline constexpr std::size_t kArchivedTensorCount = 118;
inline constexpr std::size_t kRuntimeTensorCount = 118;
inline constexpr std::string_view kSafetensorsBlobSha256{"342b52ae248f2b70b2059313bab5ae7a358cbabac940072f71f5907eeaa21536"};
inline constexpr std::string_view kSafetensorsDataSha256{"361cfe8ebe111bbfd3e006bcd57acd0acb318b3abe1583f92623607a02f6d649"};
inline constexpr std::string_view kSourceArtifactSha256{"c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"};
inline constexpr std::string_view kTensorSchema{"safetensors:proteinmpnn-v48-eps020-v1;runtime_tensors=118"};
inline constexpr std::string_view kIgnoredHistoricalTensorNames[0] = {};

inline constexpr std::size_t kTensor0Shape[] = {16, 66};
inline constexpr std::size_t kTensor0Strides[] = {66, 1};
inline constexpr std::size_t kTensor1Shape[] = {16};
inline constexpr std::size_t kTensor1Strides[] = {1};
inline constexpr std::size_t kTensor2Shape[] = {128, 416};
inline constexpr std::size_t kTensor2Strides[] = {416, 1};
inline constexpr std::size_t kTensor3Shape[] = {128};
inline constexpr std::size_t kTensor3Strides[] = {1};
inline constexpr std::size_t kTensor4Shape[] = {128};
inline constexpr std::size_t kTensor4Strides[] = {1};
inline constexpr std::size_t kTensor5Shape[] = {128, 128};
inline constexpr std::size_t kTensor5Strides[] = {128, 1};
inline constexpr std::size_t kTensor6Shape[] = {128};
inline constexpr std::size_t kTensor6Strides[] = {1};
inline constexpr std::size_t kTensor7Shape[] = {21, 128};
inline constexpr std::size_t kTensor7Strides[] = {128, 1};
inline constexpr std::size_t kTensor8Shape[] = {128};
inline constexpr std::size_t kTensor8Strides[] = {1};
inline constexpr std::size_t kTensor9Shape[] = {128};
inline constexpr std::size_t kTensor9Strides[] = {1};
inline constexpr std::size_t kTensor10Shape[] = {128};
inline constexpr std::size_t kTensor10Strides[] = {1};
inline constexpr std::size_t kTensor11Shape[] = {128};
inline constexpr std::size_t kTensor11Strides[] = {1};
inline constexpr std::size_t kTensor12Shape[] = {128};
inline constexpr std::size_t kTensor12Strides[] = {1};
inline constexpr std::size_t kTensor13Shape[] = {128};
inline constexpr std::size_t kTensor13Strides[] = {1};
inline constexpr std::size_t kTensor14Shape[] = {128, 384};
inline constexpr std::size_t kTensor14Strides[] = {384, 1};
inline constexpr std::size_t kTensor15Shape[] = {128};
inline constexpr std::size_t kTensor15Strides[] = {1};
inline constexpr std::size_t kTensor16Shape[] = {128, 128};
inline constexpr std::size_t kTensor16Strides[] = {128, 1};
inline constexpr std::size_t kTensor17Shape[] = {128};
inline constexpr std::size_t kTensor17Strides[] = {1};
inline constexpr std::size_t kTensor18Shape[] = {128, 128};
inline constexpr std::size_t kTensor18Strides[] = {128, 1};
inline constexpr std::size_t kTensor19Shape[] = {128};
inline constexpr std::size_t kTensor19Strides[] = {1};
inline constexpr std::size_t kTensor20Shape[] = {128, 384};
inline constexpr std::size_t kTensor20Strides[] = {384, 1};
inline constexpr std::size_t kTensor21Shape[] = {128};
inline constexpr std::size_t kTensor21Strides[] = {1};
inline constexpr std::size_t kTensor22Shape[] = {128, 128};
inline constexpr std::size_t kTensor22Strides[] = {128, 1};
inline constexpr std::size_t kTensor23Shape[] = {128};
inline constexpr std::size_t kTensor23Strides[] = {1};
inline constexpr std::size_t kTensor24Shape[] = {128, 128};
inline constexpr std::size_t kTensor24Strides[] = {128, 1};
inline constexpr std::size_t kTensor25Shape[] = {128};
inline constexpr std::size_t kTensor25Strides[] = {1};
inline constexpr std::size_t kTensor26Shape[] = {512, 128};
inline constexpr std::size_t kTensor26Strides[] = {128, 1};
inline constexpr std::size_t kTensor27Shape[] = {512};
inline constexpr std::size_t kTensor27Strides[] = {1};
inline constexpr std::size_t kTensor28Shape[] = {128, 512};
inline constexpr std::size_t kTensor28Strides[] = {512, 1};
inline constexpr std::size_t kTensor29Shape[] = {128};
inline constexpr std::size_t kTensor29Strides[] = {1};
inline constexpr std::size_t kTensor30Shape[] = {128};
inline constexpr std::size_t kTensor30Strides[] = {1};
inline constexpr std::size_t kTensor31Shape[] = {128};
inline constexpr std::size_t kTensor31Strides[] = {1};
inline constexpr std::size_t kTensor32Shape[] = {128};
inline constexpr std::size_t kTensor32Strides[] = {1};
inline constexpr std::size_t kTensor33Shape[] = {128};
inline constexpr std::size_t kTensor33Strides[] = {1};
inline constexpr std::size_t kTensor34Shape[] = {128};
inline constexpr std::size_t kTensor34Strides[] = {1};
inline constexpr std::size_t kTensor35Shape[] = {128};
inline constexpr std::size_t kTensor35Strides[] = {1};
inline constexpr std::size_t kTensor36Shape[] = {128, 384};
inline constexpr std::size_t kTensor36Strides[] = {384, 1};
inline constexpr std::size_t kTensor37Shape[] = {128};
inline constexpr std::size_t kTensor37Strides[] = {1};
inline constexpr std::size_t kTensor38Shape[] = {128, 128};
inline constexpr std::size_t kTensor38Strides[] = {128, 1};
inline constexpr std::size_t kTensor39Shape[] = {128};
inline constexpr std::size_t kTensor39Strides[] = {1};
inline constexpr std::size_t kTensor40Shape[] = {128, 128};
inline constexpr std::size_t kTensor40Strides[] = {128, 1};
inline constexpr std::size_t kTensor41Shape[] = {128};
inline constexpr std::size_t kTensor41Strides[] = {1};
inline constexpr std::size_t kTensor42Shape[] = {128, 384};
inline constexpr std::size_t kTensor42Strides[] = {384, 1};
inline constexpr std::size_t kTensor43Shape[] = {128};
inline constexpr std::size_t kTensor43Strides[] = {1};
inline constexpr std::size_t kTensor44Shape[] = {128, 128};
inline constexpr std::size_t kTensor44Strides[] = {128, 1};
inline constexpr std::size_t kTensor45Shape[] = {128};
inline constexpr std::size_t kTensor45Strides[] = {1};
inline constexpr std::size_t kTensor46Shape[] = {128, 128};
inline constexpr std::size_t kTensor46Strides[] = {128, 1};
inline constexpr std::size_t kTensor47Shape[] = {128};
inline constexpr std::size_t kTensor47Strides[] = {1};
inline constexpr std::size_t kTensor48Shape[] = {512, 128};
inline constexpr std::size_t kTensor48Strides[] = {128, 1};
inline constexpr std::size_t kTensor49Shape[] = {512};
inline constexpr std::size_t kTensor49Strides[] = {1};
inline constexpr std::size_t kTensor50Shape[] = {128, 512};
inline constexpr std::size_t kTensor50Strides[] = {512, 1};
inline constexpr std::size_t kTensor51Shape[] = {128};
inline constexpr std::size_t kTensor51Strides[] = {1};
inline constexpr std::size_t kTensor52Shape[] = {128};
inline constexpr std::size_t kTensor52Strides[] = {1};
inline constexpr std::size_t kTensor53Shape[] = {128};
inline constexpr std::size_t kTensor53Strides[] = {1};
inline constexpr std::size_t kTensor54Shape[] = {128};
inline constexpr std::size_t kTensor54Strides[] = {1};
inline constexpr std::size_t kTensor55Shape[] = {128};
inline constexpr std::size_t kTensor55Strides[] = {1};
inline constexpr std::size_t kTensor56Shape[] = {128};
inline constexpr std::size_t kTensor56Strides[] = {1};
inline constexpr std::size_t kTensor57Shape[] = {128};
inline constexpr std::size_t kTensor57Strides[] = {1};
inline constexpr std::size_t kTensor58Shape[] = {128, 384};
inline constexpr std::size_t kTensor58Strides[] = {384, 1};
inline constexpr std::size_t kTensor59Shape[] = {128};
inline constexpr std::size_t kTensor59Strides[] = {1};
inline constexpr std::size_t kTensor60Shape[] = {128, 128};
inline constexpr std::size_t kTensor60Strides[] = {128, 1};
inline constexpr std::size_t kTensor61Shape[] = {128};
inline constexpr std::size_t kTensor61Strides[] = {1};
inline constexpr std::size_t kTensor62Shape[] = {128, 128};
inline constexpr std::size_t kTensor62Strides[] = {128, 1};
inline constexpr std::size_t kTensor63Shape[] = {128};
inline constexpr std::size_t kTensor63Strides[] = {1};
inline constexpr std::size_t kTensor64Shape[] = {128, 384};
inline constexpr std::size_t kTensor64Strides[] = {384, 1};
inline constexpr std::size_t kTensor65Shape[] = {128};
inline constexpr std::size_t kTensor65Strides[] = {1};
inline constexpr std::size_t kTensor66Shape[] = {128, 128};
inline constexpr std::size_t kTensor66Strides[] = {128, 1};
inline constexpr std::size_t kTensor67Shape[] = {128};
inline constexpr std::size_t kTensor67Strides[] = {1};
inline constexpr std::size_t kTensor68Shape[] = {128, 128};
inline constexpr std::size_t kTensor68Strides[] = {128, 1};
inline constexpr std::size_t kTensor69Shape[] = {128};
inline constexpr std::size_t kTensor69Strides[] = {1};
inline constexpr std::size_t kTensor70Shape[] = {512, 128};
inline constexpr std::size_t kTensor70Strides[] = {128, 1};
inline constexpr std::size_t kTensor71Shape[] = {512};
inline constexpr std::size_t kTensor71Strides[] = {1};
inline constexpr std::size_t kTensor72Shape[] = {128, 512};
inline constexpr std::size_t kTensor72Strides[] = {512, 1};
inline constexpr std::size_t kTensor73Shape[] = {128};
inline constexpr std::size_t kTensor73Strides[] = {1};
inline constexpr std::size_t kTensor74Shape[] = {128};
inline constexpr std::size_t kTensor74Strides[] = {1};
inline constexpr std::size_t kTensor75Shape[] = {128};
inline constexpr std::size_t kTensor75Strides[] = {1};
inline constexpr std::size_t kTensor76Shape[] = {128};
inline constexpr std::size_t kTensor76Strides[] = {1};
inline constexpr std::size_t kTensor77Shape[] = {128};
inline constexpr std::size_t kTensor77Strides[] = {1};
inline constexpr std::size_t kTensor78Shape[] = {128, 512};
inline constexpr std::size_t kTensor78Strides[] = {512, 1};
inline constexpr std::size_t kTensor79Shape[] = {128};
inline constexpr std::size_t kTensor79Strides[] = {1};
inline constexpr std::size_t kTensor80Shape[] = {128, 128};
inline constexpr std::size_t kTensor80Strides[] = {128, 1};
inline constexpr std::size_t kTensor81Shape[] = {128};
inline constexpr std::size_t kTensor81Strides[] = {1};
inline constexpr std::size_t kTensor82Shape[] = {128, 128};
inline constexpr std::size_t kTensor82Strides[] = {128, 1};
inline constexpr std::size_t kTensor83Shape[] = {128};
inline constexpr std::size_t kTensor83Strides[] = {1};
inline constexpr std::size_t kTensor84Shape[] = {512, 128};
inline constexpr std::size_t kTensor84Strides[] = {128, 1};
inline constexpr std::size_t kTensor85Shape[] = {512};
inline constexpr std::size_t kTensor85Strides[] = {1};
inline constexpr std::size_t kTensor86Shape[] = {128, 512};
inline constexpr std::size_t kTensor86Strides[] = {512, 1};
inline constexpr std::size_t kTensor87Shape[] = {128};
inline constexpr std::size_t kTensor87Strides[] = {1};
inline constexpr std::size_t kTensor88Shape[] = {128};
inline constexpr std::size_t kTensor88Strides[] = {1};
inline constexpr std::size_t kTensor89Shape[] = {128};
inline constexpr std::size_t kTensor89Strides[] = {1};
inline constexpr std::size_t kTensor90Shape[] = {128};
inline constexpr std::size_t kTensor90Strides[] = {1};
inline constexpr std::size_t kTensor91Shape[] = {128};
inline constexpr std::size_t kTensor91Strides[] = {1};
inline constexpr std::size_t kTensor92Shape[] = {128, 512};
inline constexpr std::size_t kTensor92Strides[] = {512, 1};
inline constexpr std::size_t kTensor93Shape[] = {128};
inline constexpr std::size_t kTensor93Strides[] = {1};
inline constexpr std::size_t kTensor94Shape[] = {128, 128};
inline constexpr std::size_t kTensor94Strides[] = {128, 1};
inline constexpr std::size_t kTensor95Shape[] = {128};
inline constexpr std::size_t kTensor95Strides[] = {1};
inline constexpr std::size_t kTensor96Shape[] = {128, 128};
inline constexpr std::size_t kTensor96Strides[] = {128, 1};
inline constexpr std::size_t kTensor97Shape[] = {128};
inline constexpr std::size_t kTensor97Strides[] = {1};
inline constexpr std::size_t kTensor98Shape[] = {512, 128};
inline constexpr std::size_t kTensor98Strides[] = {128, 1};
inline constexpr std::size_t kTensor99Shape[] = {512};
inline constexpr std::size_t kTensor99Strides[] = {1};
inline constexpr std::size_t kTensor100Shape[] = {128, 512};
inline constexpr std::size_t kTensor100Strides[] = {512, 1};
inline constexpr std::size_t kTensor101Shape[] = {128};
inline constexpr std::size_t kTensor101Strides[] = {1};
inline constexpr std::size_t kTensor102Shape[] = {128};
inline constexpr std::size_t kTensor102Strides[] = {1};
inline constexpr std::size_t kTensor103Shape[] = {128};
inline constexpr std::size_t kTensor103Strides[] = {1};
inline constexpr std::size_t kTensor104Shape[] = {128};
inline constexpr std::size_t kTensor104Strides[] = {1};
inline constexpr std::size_t kTensor105Shape[] = {128};
inline constexpr std::size_t kTensor105Strides[] = {1};
inline constexpr std::size_t kTensor106Shape[] = {128, 512};
inline constexpr std::size_t kTensor106Strides[] = {512, 1};
inline constexpr std::size_t kTensor107Shape[] = {128};
inline constexpr std::size_t kTensor107Strides[] = {1};
inline constexpr std::size_t kTensor108Shape[] = {128, 128};
inline constexpr std::size_t kTensor108Strides[] = {128, 1};
inline constexpr std::size_t kTensor109Shape[] = {128};
inline constexpr std::size_t kTensor109Strides[] = {1};
inline constexpr std::size_t kTensor110Shape[] = {128, 128};
inline constexpr std::size_t kTensor110Strides[] = {128, 1};
inline constexpr std::size_t kTensor111Shape[] = {128};
inline constexpr std::size_t kTensor111Strides[] = {1};
inline constexpr std::size_t kTensor112Shape[] = {512, 128};
inline constexpr std::size_t kTensor112Strides[] = {128, 1};
inline constexpr std::size_t kTensor113Shape[] = {512};
inline constexpr std::size_t kTensor113Strides[] = {1};
inline constexpr std::size_t kTensor114Shape[] = {128, 512};
inline constexpr std::size_t kTensor114Strides[] = {512, 1};
inline constexpr std::size_t kTensor115Shape[] = {128};
inline constexpr std::size_t kTensor115Strides[] = {1};
inline constexpr std::size_t kTensor116Shape[] = {21, 128};
inline constexpr std::size_t kTensor116Strides[] = {128, 1};
inline constexpr std::size_t kTensor117Shape[] = {21};
inline constexpr std::size_t kTensor117Strides[] = {1};

inline constexpr TensorBlobInfo kRuntimeTensors[] = {
    {"features.embeddings.linear.weight", 6636692, 4224, kTensor0Shape, kTensor0Strides, 2, "float32", "65d027dc3d7c195529916c89e8ff191e051f7c812458626623405e43ccab41f9"},
    {"features.embeddings.linear.bias", 6636628, 64, kTensor1Shape, kTensor1Strides, 1, "float32", "050324527a64580e5e979a133c27d94ba2e14600348ecedb010ee244dc3a97ef"},
    {"features.edge_embedding.weight", 6423636, 212992, kTensor2Shape, kTensor2Strides, 2, "float32", "4016c21e7c83fae37856604e517aa28a8a7367e6ca1f965065962af57fb24ff7"},
    {"features.norm_edges.weight", 6641428, 512, kTensor3Shape, kTensor3Strides, 1, "float32", "7b6a0d8dbf740533b365e942d5774bbb83d2f85c9217bd0c96f1bfdacd091189"},
    {"features.norm_edges.bias", 6640916, 512, kTensor4Shape, kTensor4Strides, 1, "float32", "d8056320bbdd5e60fe0b6865beda6451bb33064e1d4c9b28b5a8f63e54b282dd"},
    {"W_e.weight", 512, 65536, kTensor5Shape, kTensor5Strides, 2, "float32", "3258a38d81d608518e2f7980129b8fcf3e1c24f35edede9d8cc86d158f4ec726"},
    {"W_e.bias", 0, 512, kTensor6Shape, kTensor6Strides, 1, "float32", "64dec33c7d79cf26cbca7b63d22d8cfbc3826a53353eca1bbb420510ddfe8c72"},
    {"W_s.weight", 76884, 10752, kTensor7Shape, kTensor7Strides, 2, "float32", "419b47e12874e74d472cd10ab844bda5d560ed3553441111b4f1440375ca8ebd"},
    {"encoder_layers.0.norm1.weight", 4044372, 512, kTensor8Shape, kTensor8Strides, 1, "float32", "c482578972130c16dc13114ada5f54f15cbd1d5d3369dd19ceafbef4e7b49b80"},
    {"encoder_layers.0.norm1.bias", 4043860, 512, kTensor9Shape, kTensor9Strides, 1, "float32", "7e71783083b13fed789cb37d8f0e1504b70ff3b7a51b20240fac1044010dc292"},
    {"encoder_layers.0.norm2.weight", 4045396, 512, kTensor10Shape, kTensor10Strides, 1, "float32", "2b24141c56a4e4a40b2d7d8d1adde68299c5e4e4e1160adb8a3c1da6c7c627e4"},
    {"encoder_layers.0.norm2.bias", 4044884, 512, kTensor11Shape, kTensor11Strides, 1, "float32", "472ed6ffb992064cf20025f188c310c1dd8f5115ddcd3ad1c3a8d9153fad406c"},
    {"encoder_layers.0.norm3.weight", 4046420, 512, kTensor12Shape, kTensor12Strides, 1, "float32", "0550a81e9f198dbbe2524738d7bb30b39787daf3503d848462a75347bfa02551"},
    {"encoder_layers.0.norm3.bias", 4045908, 512, kTensor13Shape, kTensor13Strides, 1, "float32", "975f9b4b3c6a65ac1d1ea3b362b8501af1e88b94f6d37e3a7434319f3bbd1da1"},
    {"encoder_layers.0.W1.weight", 2859092, 196608, kTensor14Shape, kTensor14Strides, 2, "float32", "a9757d7aca5e55c43e9fc994e559c6d92704946d40c0f49555c6e2a91ad95519"},
    {"encoder_layers.0.W1.bias", 2858580, 512, kTensor15Shape, kTensor15Strides, 1, "float32", "80ac64b62343b7fe8baa6e4dbf33c022349c98631d18b81497dbe6a3cf327176"},
    {"encoder_layers.0.W2.weight", 3385428, 65536, kTensor16Shape, kTensor16Strides, 2, "float32", "7fa914c133af10142a6b06ad4d34bd2c50b506cf7e9a93fe5933b3492d9e9f38"},
    {"encoder_layers.0.W2.bias", 3384916, 512, kTensor17Shape, kTensor17Strides, 1, "float32", "0fb8080278eff3124fd683d69edd40e8b25b5fe6ef890180261d714b77ecd677"},
    {"encoder_layers.0.W3.weight", 3451476, 65536, kTensor18Shape, kTensor18Strides, 2, "float32", "94e5adff368a6ff1cab397b21cd3b132e80ed7d4760b5eb7a4ceefb49246ab3b"},
    {"encoder_layers.0.W3.bias", 3450964, 512, kTensor19Shape, kTensor19Strides, 1, "float32", "2773c49503d553b40567b8ce0742104b0495d352b41db034aa2f793367744b92"},
    {"encoder_layers.0.W11.weight", 3056212, 196608, kTensor20Shape, kTensor20Strides, 2, "float32", "dd00ccca8663144ea7de22ecf5b6c56d429237c095b0cd8d17eb3d711b8dbbff"},
    {"encoder_layers.0.W11.bias", 3055700, 512, kTensor21Shape, kTensor21Strides, 1, "float32", "a499b6980d40765121843208659dd8c3e967d9e0ae63cc58638c65f70b3d3973"},
    {"encoder_layers.0.W12.weight", 3253332, 65536, kTensor22Shape, kTensor22Strides, 2, "float32", "3b3cba9af09ce1a5327ffa567fba9ef93d06e684005e26cf1bd4548f8d53abd2"},
    {"encoder_layers.0.W12.bias", 3252820, 512, kTensor23Shape, kTensor23Strides, 1, "float32", "ee9525c4a011e0d0129052a89bdfd7ea85dd1fc0f39ff20c1bbd00dd9eaafcd6"},
    {"encoder_layers.0.W13.weight", 3319380, 65536, kTensor24Shape, kTensor24Strides, 2, "float32", "6c40d13a493a4649855d62f6053449670d05c70e7a523a09ef7559797c6664d2"},
    {"encoder_layers.0.W13.bias", 3318868, 512, kTensor25Shape, kTensor25Strides, 1, "float32", "d85a371b849db87370923889caa67a1bc5f7d3de1d99aab456cea949dd7b6009"},
    {"encoder_layers.0.dense.W_in.weight", 3519060, 262144, kTensor26Shape, kTensor26Strides, 2, "float32", "0aea6f865870120883473cbee1fe276d09a2018c8c21e1a0cc71a5e62e9a6ff6"},
    {"encoder_layers.0.dense.W_in.bias", 3517012, 2048, kTensor27Shape, kTensor27Strides, 1, "float32", "e637148438a77ac631c8c3ea630f7493bf16be6d5a5b715fc96be6f5edc75fd8"},
    {"encoder_layers.0.dense.W_out.weight", 3781716, 262144, kTensor28Shape, kTensor28Strides, 2, "float32", "5bcd1570cc22064f034f0b8c2ad51cbf7e5c5971e52798b6e4674b23b122e7dd"},
    {"encoder_layers.0.dense.W_out.bias", 3781204, 512, kTensor29Shape, kTensor29Strides, 1, "float32", "b05f3f93b93f2a7229eab086200e1373f5488d0579754fd3c9fff7c1ea5dffde"},
    {"encoder_layers.1.norm1.weight", 5232724, 512, kTensor30Shape, kTensor30Strides, 1, "float32", "3fbe9cb2f112a9bf67d5308e9ad06064b14bad3fc31168e9362e978340055eb6"},
    {"encoder_layers.1.norm1.bias", 5232212, 512, kTensor31Shape, kTensor31Strides, 1, "float32", "c9b5dd4631cf16351ca1e5694d8aa69028be241a7e431a36b817451acf28ccaa"},
    {"encoder_layers.1.norm2.weight", 5233748, 512, kTensor32Shape, kTensor32Strides, 1, "float32", "8e485b891872fb7a95e79a351486e063e82a9a6d738e9b64eee8f3c2e5e49857"},
    {"encoder_layers.1.norm2.bias", 5233236, 512, kTensor33Shape, kTensor33Strides, 1, "float32", "7f25927417c353b24010c76fcf9e5866e4eb1b4e50bbe176e646d94934a50347"},
    {"encoder_layers.1.norm3.weight", 5234772, 512, kTensor34Shape, kTensor34Strides, 1, "float32", "1477031529ed4d4c401fb952f8a30a3e54fffe354f76a1b6db35ceeda1f93923"},
    {"encoder_layers.1.norm3.bias", 5234260, 512, kTensor35Shape, kTensor35Strides, 1, "float32", "db73adec6f2d63fc638a50ee96933b6e8cc7648d5cbd73277ada1ee6841160e3"},
    {"encoder_layers.1.W1.weight", 4047444, 196608, kTensor36Shape, kTensor36Strides, 2, "float32", "bb4c2cec290f851abb927a3871d99946940af29233058f3066a712c77ea6d42f"},
    {"encoder_layers.1.W1.bias", 4046932, 512, kTensor37Shape, kTensor37Strides, 1, "float32", "0a3b60e46fbf1777fab047ae96d0d17d42a0a3162a3e6b4ced4df7402108a997"},
    {"encoder_layers.1.W2.weight", 4573780, 65536, kTensor38Shape, kTensor38Strides, 2, "float32", "0f0be1301c94c97d1eef732b02c28de0127c003ff5d8bd8a2011b9f6029d61c9"},
    {"encoder_layers.1.W2.bias", 4573268, 512, kTensor39Shape, kTensor39Strides, 1, "float32", "6ef71de74cf03857d1b949414e5165a839dbafc864f7af0886c7f8b2f93b2828"},
    {"encoder_layers.1.W3.weight", 4639828, 65536, kTensor40Shape, kTensor40Strides, 2, "float32", "58be6eff93c69e7ad7f93a3ea1bd59c3e19f188e72ad6fb95134a3a8bcc7f1dc"},
    {"encoder_layers.1.W3.bias", 4639316, 512, kTensor41Shape, kTensor41Strides, 1, "float32", "5c6452ae8ea802631853e705ada056c96d0a8420da66bd2295586bd922a4debf"},
    {"encoder_layers.1.W11.weight", 4244564, 196608, kTensor42Shape, kTensor42Strides, 2, "float32", "432d478d4232fbfc966e064fde8ac365a1ddff8c2291548c0fee720e604f2ccf"},
    {"encoder_layers.1.W11.bias", 4244052, 512, kTensor43Shape, kTensor43Strides, 1, "float32", "399250ea1b1bd3639e3f28fd513097081aa733f18229122e85020307c323b689"},
    {"encoder_layers.1.W12.weight", 4441684, 65536, kTensor44Shape, kTensor44Strides, 2, "float32", "2ec4b0cf7ef816fe71e98dc5b3db1233dd48d29d2153332eb89638ad3ed50b03"},
    {"encoder_layers.1.W12.bias", 4441172, 512, kTensor45Shape, kTensor45Strides, 1, "float32", "518f03f47e88279f2ef83e548911283df03fd908bbd6b6df432b0bedcb4efc75"},
    {"encoder_layers.1.W13.weight", 4507732, 65536, kTensor46Shape, kTensor46Strides, 2, "float32", "9e23ef8f5fb6154916af47a364a612491694dacf974f007222e74dd4daea3336"},
    {"encoder_layers.1.W13.bias", 4507220, 512, kTensor47Shape, kTensor47Strides, 1, "float32", "e232e90a14476a3cf02b3b9c11a2db4fb1051e02e477966ae558fc3b642bbb9f"},
    {"encoder_layers.1.dense.W_in.weight", 4707412, 262144, kTensor48Shape, kTensor48Strides, 2, "float32", "a0cfd82d0d9fb291a7856b23dfec0737172b29a9f13c3acc75b5082bdf37afaf"},
    {"encoder_layers.1.dense.W_in.bias", 4705364, 2048, kTensor49Shape, kTensor49Strides, 1, "float32", "8f95f84c746df7a22cde99f589e460507a79f4fcbba73cd3c21b99d09f33b05c"},
    {"encoder_layers.1.dense.W_out.weight", 4970068, 262144, kTensor50Shape, kTensor50Strides, 2, "float32", "c8d4a50546d00dbba29955a6df4487b1a5fda09b507bb220bc5d9fd21ae3917d"},
    {"encoder_layers.1.dense.W_out.bias", 4969556, 512, kTensor51Shape, kTensor51Strides, 1, "float32", "6e9ceb62c13042c9f507c00f28dc3751a59e4e0eb80b2df73694bd399880649b"},
    {"encoder_layers.2.norm1.weight", 6421076, 512, kTensor52Shape, kTensor52Strides, 1, "float32", "87e1134fcd496a74a39e731f20cef4d3bcbac648315a8fe7f4ac43c6a3524e5d"},
    {"encoder_layers.2.norm1.bias", 6420564, 512, kTensor53Shape, kTensor53Strides, 1, "float32", "b01c471841f8e3127300f99af021b6a96f39d5f603f70ad862dc83bca57f370c"},
    {"encoder_layers.2.norm2.weight", 6422100, 512, kTensor54Shape, kTensor54Strides, 1, "float32", "70d5d717110d9e64a0a321b2c2bda33d4af0fc315a0c0beaf09ef9b2d46a0c9e"},
    {"encoder_layers.2.norm2.bias", 6421588, 512, kTensor55Shape, kTensor55Strides, 1, "float32", "636eacc2ea39ba64f226219999c31f8017574f8691fb1af904d6e90662d8e44c"},
    {"encoder_layers.2.norm3.weight", 6423124, 512, kTensor56Shape, kTensor56Strides, 1, "float32", "8f583e2d6ba86a7dbfe4bd9e06c87ca3074be0ebb9127dda6be7858f95fecb17"},
    {"encoder_layers.2.norm3.bias", 6422612, 512, kTensor57Shape, kTensor57Strides, 1, "float32", "43a5b03ffa37ee768d2627e284a41c366de8d1f079d0f123357a9c77dc32ec3a"},
    {"encoder_layers.2.W1.weight", 5235796, 196608, kTensor58Shape, kTensor58Strides, 2, "float32", "f1c2d7ed05c72aa1a44847da73c10203fae39c86c183e718c023e5cfa6867cae"},
    {"encoder_layers.2.W1.bias", 5235284, 512, kTensor59Shape, kTensor59Strides, 1, "float32", "18c539bd9628c36646d0f3113864b18986dddb42a81f4fa8d296358969e8ca91"},
    {"encoder_layers.2.W2.weight", 5762132, 65536, kTensor60Shape, kTensor60Strides, 2, "float32", "3099027e07da4d7d8911a7993455fba9484595a6f20497d189798ac7edeecb09"},
    {"encoder_layers.2.W2.bias", 5761620, 512, kTensor61Shape, kTensor61Strides, 1, "float32", "5fc52d0f8fcf9e1c222d3a719e513233c7c9ddd9a3904a853636116577bccc9e"},
    {"encoder_layers.2.W3.weight", 5828180, 65536, kTensor62Shape, kTensor62Strides, 2, "float32", "7d47153fa4bd47d47be9fc55d0a4091b26b2dc865d4100085121a9cb423b879f"},
    {"encoder_layers.2.W3.bias", 5827668, 512, kTensor63Shape, kTensor63Strides, 1, "float32", "3e8149e7ba543907b6232ad1d28f642c136786d26283a92a9db8a56e7d0e7132"},
    {"encoder_layers.2.W11.weight", 5432916, 196608, kTensor64Shape, kTensor64Strides, 2, "float32", "602f1e001499e7716cc6a552808edc1a980b276a62ced42f5c21e0efd09d8544"},
    {"encoder_layers.2.W11.bias", 5432404, 512, kTensor65Shape, kTensor65Strides, 1, "float32", "485163845ecd7e46f86552089d8525dd1ea534b1ffb74f19e913d59a830c2a2a"},
    {"encoder_layers.2.W12.weight", 5630036, 65536, kTensor66Shape, kTensor66Strides, 2, "float32", "ce05bdffcdde3395a9fe211b897c6baaa578fed5056bd8cda0c81100a1169c0c"},
    {"encoder_layers.2.W12.bias", 5629524, 512, kTensor67Shape, kTensor67Strides, 1, "float32", "5c933655b57e31055e2b2de55798a4f8627f07adfa985031c21ef061c2584099"},
    {"encoder_layers.2.W13.weight", 5696084, 65536, kTensor68Shape, kTensor68Strides, 2, "float32", "23961ea5408cdce4d8d6f7a2b3e13e8b110aaede95a160f025e8cbce93b957e2"},
    {"encoder_layers.2.W13.bias", 5695572, 512, kTensor69Shape, kTensor69Strides, 1, "float32", "7b583eab784143f821e438a901b4644f24f1810bd13ecd661a24387e008810e7"},
    {"encoder_layers.2.dense.W_in.weight", 5895764, 262144, kTensor70Shape, kTensor70Strides, 2, "float32", "9de372c8ff58a90860f35455cf75a69c10ebb7558f9ec5188b948b35c6ff1a75"},
    {"encoder_layers.2.dense.W_in.bias", 5893716, 2048, kTensor71Shape, kTensor71Strides, 1, "float32", "4942ef011d8dfe0b263d6b7e34ccccb77fd77ad04b667a352126a2e12cc9e1a9"},
    {"encoder_layers.2.dense.W_out.weight", 6158420, 262144, kTensor72Shape, kTensor72Strides, 2, "float32", "242bd83f28017f71090063cb519bf1cae010839419af1baecbbc45ce1d061c0d"},
    {"encoder_layers.2.dense.W_out.bias", 6157908, 512, kTensor73Shape, kTensor73Strides, 1, "float32", "1a7bac5fc0ebd62ce51396d0cc3e92f35f8b592f8352df693f74511873e71054"},
    {"decoder_layers.0.norm1.weight", 1009748, 512, kTensor74Shape, kTensor74Strides, 1, "float32", "982024cb02329064987935c9911c98c4325fe1679bd611383c0b08b20f247e96"},
    {"decoder_layers.0.norm1.bias", 1009236, 512, kTensor75Shape, kTensor75Strides, 1, "float32", "f82494e3ca4bc72a9d0a9f86eb7042ffc262683979803ee5ad2e086d5ffb7a10"},
    {"decoder_layers.0.norm2.weight", 1010772, 512, kTensor76Shape, kTensor76Strides, 1, "float32", "097979b4b1c3589a6aabef4420976cc0df1eeba7414ec5e432e11e923f44b7b0"},
    {"decoder_layers.0.norm2.bias", 1010260, 512, kTensor77Shape, kTensor77Strides, 1, "float32", "4879b24bf7d858dc3402af35a112ebd38859c0da166f39ab11942b58f07a617c"},
    {"decoder_layers.0.W1.weight", 88148, 262144, kTensor78Shape, kTensor78Strides, 2, "float32", "3416982523cb1f5bb15f7d2b0a56ae6c923c6b3ad2983b719a2ee258775e908b"},
    {"decoder_layers.0.W1.bias", 87636, 512, kTensor79Shape, kTensor79Strides, 1, "float32", "eb9b5f71fbe664eaa45052c542fe34ce2f6f9d17a7d64d851e3a018e387136ec"},
    {"decoder_layers.0.W2.weight", 350804, 65536, kTensor80Shape, kTensor80Strides, 2, "float32", "1cf72744343b3fb10d5266cf3dfb758e75be85c9d8f12a8012bdbd8b6377bea6"},
    {"decoder_layers.0.W2.bias", 350292, 512, kTensor81Shape, kTensor81Strides, 1, "float32", "f4e1c875b95be802dfa6791c8d8a74c196f9bc03300aff73c471be5cb7a8332e"},
    {"decoder_layers.0.W3.weight", 416852, 65536, kTensor82Shape, kTensor82Strides, 2, "float32", "692a191f6d23f713060b027040e03292024f3c25726966ea66303f769975069a"},
    {"decoder_layers.0.W3.bias", 416340, 512, kTensor83Shape, kTensor83Strides, 1, "float32", "f9e5b04ef488079b400af31851073095fd2bd63601f472d35c9cda5d77260d20"},
    {"decoder_layers.0.dense.W_in.weight", 484436, 262144, kTensor84Shape, kTensor84Strides, 2, "float32", "74f34f32d561b6c57901970da96c29d48e4dccd97d0fbc6a553273197b1d9f87"},
    {"decoder_layers.0.dense.W_in.bias", 482388, 2048, kTensor85Shape, kTensor85Strides, 1, "float32", "9d33b73603e747e62d4275deba81e475ee42b1963b08eb1fd91bc4473222e8ed"},
    {"decoder_layers.0.dense.W_out.weight", 747092, 262144, kTensor86Shape, kTensor86Strides, 2, "float32", "7985b954bbe91257242487577d7381d0956f550fcdac971f8f022173d1775daf"},
    {"decoder_layers.0.dense.W_out.bias", 746580, 512, kTensor87Shape, kTensor87Strides, 1, "float32", "818d164459440260533428d8e3ee3ef5135094b23f06f0147c0d23362fe2d638"},
    {"decoder_layers.1.norm1.weight", 1933396, 512, kTensor88Shape, kTensor88Strides, 1, "float32", "1e008bedd431d36bd1b9d3f76df0f5423d2e82aadf10415eb880b2804ea9bac6"},
    {"decoder_layers.1.norm1.bias", 1932884, 512, kTensor89Shape, kTensor89Strides, 1, "float32", "b0aa1f5eb85e4d5ca1237ecc5d8eb9712d2e6372fff10bfc0e3720ae765176bb"},
    {"decoder_layers.1.norm2.weight", 1934420, 512, kTensor90Shape, kTensor90Strides, 1, "float32", "5938d1c744623c229782e6d1859ec636f3a5933f7d44d4bf6f5ffe649dbc9b4e"},
    {"decoder_layers.1.norm2.bias", 1933908, 512, kTensor91Shape, kTensor91Strides, 1, "float32", "98038445ac02bd9eff0b617b516f33154d0407659689e8df94158c0b84dd205b"},
    {"decoder_layers.1.W1.weight", 1011796, 262144, kTensor92Shape, kTensor92Strides, 2, "float32", "c6ae54db65849b94efb0263544d903d8945a3c191e12ed84dbd555c49cff9e0c"},
    {"decoder_layers.1.W1.bias", 1011284, 512, kTensor93Shape, kTensor93Strides, 1, "float32", "cb6f982c55696098ca2cebbe456fdaa6d18ae45f4fbe41414629384201c4a26b"},
    {"decoder_layers.1.W2.weight", 1274452, 65536, kTensor94Shape, kTensor94Strides, 2, "float32", "11043b23b3b9b2c2e1c62ec5a2b5cb2b68b9d875bb6f21aabccd62aea4871f02"},
    {"decoder_layers.1.W2.bias", 1273940, 512, kTensor95Shape, kTensor95Strides, 1, "float32", "f7fde1ffaeeb401f481a5fe72737a24e08c7fc8e0263440bd8704d8f8a5fa603"},
    {"decoder_layers.1.W3.weight", 1340500, 65536, kTensor96Shape, kTensor96Strides, 2, "float32", "e96d5c6e399ac40266ffaf1d1a9d48ea4620a5d8abf72e3fbd512ed23e496e0f"},
    {"decoder_layers.1.W3.bias", 1339988, 512, kTensor97Shape, kTensor97Strides, 1, "float32", "fb67e157a961407f70c2c5d922978f138d514daafa78c59c230665df675142c2"},
    {"decoder_layers.1.dense.W_in.weight", 1408084, 262144, kTensor98Shape, kTensor98Strides, 2, "float32", "625cd057715d56a066bfd4737e7066049a2625ae82d1eb05b1001e05c9a77a80"},
    {"decoder_layers.1.dense.W_in.bias", 1406036, 2048, kTensor99Shape, kTensor99Strides, 1, "float32", "d24fac6f5791455eedff750bf7367823ea438f208dcd061e5e300a61f1473d4b"},
    {"decoder_layers.1.dense.W_out.weight", 1670740, 262144, kTensor100Shape, kTensor100Strides, 2, "float32", "c71702d8324a0d9a42ccf1b0b1b9f4a6d730bcf9c982580be3dadb852db90436"},
    {"decoder_layers.1.dense.W_out.bias", 1670228, 512, kTensor101Shape, kTensor101Strides, 1, "float32", "50c75a028c539f16d0d5a4a0b95ae40bea5703682156c9a210bc18ccceba827a"},
    {"decoder_layers.2.norm1.weight", 2857044, 512, kTensor102Shape, kTensor102Strides, 1, "float32", "b92df0eb649b9f3c0eb9f50209e73c35fa36a6d7448eb223076a54787c38975a"},
    {"decoder_layers.2.norm1.bias", 2856532, 512, kTensor103Shape, kTensor103Strides, 1, "float32", "129196c625a7c22d00f01b4f231944ae115cfd981a71dc5390536f61cac10a02"},
    {"decoder_layers.2.norm2.weight", 2858068, 512, kTensor104Shape, kTensor104Strides, 1, "float32", "81021569feabaadf47a61d6f5ccf3cb5afbbee552e997567d64efb597bc3f7a8"},
    {"decoder_layers.2.norm2.bias", 2857556, 512, kTensor105Shape, kTensor105Strides, 1, "float32", "4b99da6c0ebafd7f82dba08e6550c58f589a0a62590a7da3490f218bcdbede4e"},
    {"decoder_layers.2.W1.weight", 1935444, 262144, kTensor106Shape, kTensor106Strides, 2, "float32", "b8aad88f52a1b441ec00ef999a2feee9498e783218265e45d0222ef9a0b0ec28"},
    {"decoder_layers.2.W1.bias", 1934932, 512, kTensor107Shape, kTensor107Strides, 1, "float32", "45c0e5aa259419c9741e8340e2217fa0d51b452cf33c827016ac253c8dd50b8b"},
    {"decoder_layers.2.W2.weight", 2198100, 65536, kTensor108Shape, kTensor108Strides, 2, "float32", "1e449b5f18d9449dc8c016d49c26e5fba741fcbd77cd59431c70e0817123b3bb"},
    {"decoder_layers.2.W2.bias", 2197588, 512, kTensor109Shape, kTensor109Strides, 1, "float32", "731453052d8da7a5173e585af82763fea1db5f20bd6a3e58dda45d9cc846a7c1"},
    {"decoder_layers.2.W3.weight", 2264148, 65536, kTensor110Shape, kTensor110Strides, 2, "float32", "5046f1aa16051f2afbb53c8b637d6781b6acf04ff7b9097d63e33969cc825764"},
    {"decoder_layers.2.W3.bias", 2263636, 512, kTensor111Shape, kTensor111Strides, 1, "float32", "3558a9bc1d3b1dc8ea6ad71a6f7774fc46718db54da9451342d3f69e7a282acd"},
    {"decoder_layers.2.dense.W_in.weight", 2331732, 262144, kTensor112Shape, kTensor112Strides, 2, "float32", "cd0566b1bd4e498efbe0c69251eb8860bbf9a44b8b047da3ecb1cc682c7cc393"},
    {"decoder_layers.2.dense.W_in.bias", 2329684, 2048, kTensor113Shape, kTensor113Strides, 1, "float32", "2fb1f2857491d8c7da32e98064a1e3acfc2aef8402cb61964edcc93d2cfcc450"},
    {"decoder_layers.2.dense.W_out.weight", 2594388, 262144, kTensor114Shape, kTensor114Strides, 2, "float32", "d08ba37193ace54c5e586f217713a18ef9648d043587ef9e7bc1c6665126fc42"},
    {"decoder_layers.2.dense.W_out.bias", 2593876, 512, kTensor115Shape, kTensor115Strides, 1, "float32", "749364bf4a0c42f9f72d697fb3cff7b9a56ee208d1a5f75729e0205fe9cdece2"},
    {"W_out.weight", 66132, 10752, kTensor116Shape, kTensor116Strides, 2, "float32", "35b0b42d053562b26de5b8d76ad8e9cc6ca3ecd1e7c51731105ad7de603f00e2"},
    {"W_out.bias", 66048, 84, kTensor117Shape, kTensor117Strides, 1, "float32", "5474231da5c600ca696b0c0d88b9ff179e0a106d5d93625fc3e7aee9daf2618a"},
};

alignas(64) extern const std::uint8_t kSafetensorsBlob[kSafetensorsBlobLength];

static_assert(sizeof(kSafetensorsBlob) == kSafetensorsBlobLength,
              "proteinmpnn-v48-eps020 blob length mismatch");
static_assert(kSafetensorsDataOffset + kSafetensorsDataLength ==
                  kSafetensorsBlobLength,
              "proteinmpnn-v48-eps020 safetensors layout mismatch");

}  // namespace hikoboshi::weights::generated::proteinmpnn_v48_eps020

#endif  // HIKOBOSHI_WEIGHTS_GENERATED_PROTEINMPNN_V48_EPS020_BLOB_HPP
