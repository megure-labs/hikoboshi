#include <hikoboshi/weights/manifest.hpp>

#include "generated/proteinmpnn_v48_eps020_blob.hpp"

#include <array>

namespace hikoboshi::weights {
namespace {

constexpr std::size_t kTensor0Shape[] = {64};
constexpr std::size_t kTensor1Shape[] = {64, 64};
constexpr std::size_t kTensor2Shape[] = {64};
constexpr std::size_t kTensor3Shape[] = {64};
constexpr std::size_t kTensor4Shape[] = {64, 416};
constexpr std::size_t kTensor5Shape[] = {64};
constexpr std::size_t kTensor6Shape[] = {64, 192};
constexpr std::size_t kTensor7Shape[] = {64};
constexpr std::size_t kTensor8Shape[] = {64, 192};
constexpr std::size_t kTensor9Shape[] = {64};
constexpr std::size_t kTensor10Shape[] = {64, 64};
constexpr std::size_t kTensor11Shape[] = {64};
constexpr std::size_t kTensor12Shape[] = {64, 64};
constexpr std::size_t kTensor13Shape[] = {64};
constexpr std::size_t kTensor14Shape[] = {64, 64};
constexpr std::size_t kTensor15Shape[] = {64};
constexpr std::size_t kTensor16Shape[] = {64, 64};
constexpr std::size_t kTensor17Shape[] = {256};
constexpr std::size_t kTensor18Shape[] = {256, 64};
constexpr std::size_t kTensor19Shape[] = {64};
constexpr std::size_t kTensor20Shape[] = {64, 256};
constexpr std::size_t kTensor21Shape[] = {64};
constexpr std::size_t kTensor22Shape[] = {64};
constexpr std::size_t kTensor23Shape[] = {64};
constexpr std::size_t kTensor24Shape[] = {64};
constexpr std::size_t kTensor25Shape[] = {64};
constexpr std::size_t kTensor26Shape[] = {64};
constexpr std::size_t kTensor27Shape[] = {64};
constexpr std::size_t kTensor28Shape[] = {64, 192};
constexpr std::size_t kTensor29Shape[] = {64};
constexpr std::size_t kTensor30Shape[] = {64, 192};
constexpr std::size_t kTensor31Shape[] = {64};
constexpr std::size_t kTensor32Shape[] = {64, 64};
constexpr std::size_t kTensor33Shape[] = {64};
constexpr std::size_t kTensor34Shape[] = {64, 64};
constexpr std::size_t kTensor35Shape[] = {64};
constexpr std::size_t kTensor36Shape[] = {64, 64};
constexpr std::size_t kTensor37Shape[] = {64};
constexpr std::size_t kTensor38Shape[] = {64, 64};
constexpr std::size_t kTensor39Shape[] = {256};
constexpr std::size_t kTensor40Shape[] = {256, 64};
constexpr std::size_t kTensor41Shape[] = {64};
constexpr std::size_t kTensor42Shape[] = {64, 256};
constexpr std::size_t kTensor43Shape[] = {64};
constexpr std::size_t kTensor44Shape[] = {64};
constexpr std::size_t kTensor45Shape[] = {64};
constexpr std::size_t kTensor46Shape[] = {64};
constexpr std::size_t kTensor47Shape[] = {64};
constexpr std::size_t kTensor48Shape[] = {64};
constexpr std::size_t kTensor49Shape[] = {64};
constexpr std::size_t kTensor50Shape[] = {64, 192};
constexpr std::size_t kTensor51Shape[] = {64};
constexpr std::size_t kTensor52Shape[] = {64, 192};
constexpr std::size_t kTensor53Shape[] = {64};
constexpr std::size_t kTensor54Shape[] = {64, 64};
constexpr std::size_t kTensor55Shape[] = {64};
constexpr std::size_t kTensor56Shape[] = {64, 64};
constexpr std::size_t kTensor57Shape[] = {64};
constexpr std::size_t kTensor58Shape[] = {64, 64};
constexpr std::size_t kTensor59Shape[] = {64};
constexpr std::size_t kTensor60Shape[] = {64, 64};
constexpr std::size_t kTensor61Shape[] = {256};
constexpr std::size_t kTensor62Shape[] = {256, 64};
constexpr std::size_t kTensor63Shape[] = {64};
constexpr std::size_t kTensor64Shape[] = {64, 256};
constexpr std::size_t kTensor65Shape[] = {64};
constexpr std::size_t kTensor66Shape[] = {64};
constexpr std::size_t kTensor67Shape[] = {64};
constexpr std::size_t kTensor68Shape[] = {64};
constexpr std::size_t kTensor69Shape[] = {64};
constexpr std::size_t kTensor70Shape[] = {64};
constexpr std::size_t kTensor71Shape[] = {16};
constexpr std::size_t kTensor72Shape[] = {16, 66};

constexpr TensorManifestView kTensorManifests[] = {
    {"W_e.bias", {kTensor0Shape, 1 }, "float32", "31c85828f5209ebe2ac2bef421c4cd32b00ffb7d59c525238d0a49396b8ab878"},
    {"W_e.weight", {kTensor1Shape, 2 }, "float32", "ef415d5629a1ec8b3dacc4bc7bc06399c91052e0e96c067209b72b7a18b4a44d"},
    {"edge_embedding.norm.bias", {kTensor2Shape, 1 }, "float32", "a217449b93fe3cce4b7c69fc00f9c10972261d9a159c3fb85ae0269099fa1b85"},
    {"edge_embedding.norm.weight", {kTensor3Shape, 1 }, "float32", "73012bf5eac210d02e008eb019e4ae035f9d75892b570612e2b880fa75f01306"},
    {"edge_embedding.weight", {kTensor4Shape, 2 }, "float32", "367cedcf9ff839abb313d74f2470b4cfa52dac49da693a533ef3afb776527f19"},
    {"layers.0.W1.bias", {kTensor5Shape, 1 }, "float32", "e8b2b4244909357f7370045446d5dad4a1a17ce24a603598590594a594e308be"},
    {"layers.0.W1.weight", {kTensor6Shape, 2 }, "float32", "d1a045b9013d884456cb0bfc198496054df04c6a0300bf23617a5d74d304139d"},
    {"layers.0.W11.bias", {kTensor7Shape, 1 }, "float32", "0133050d58103df51e57f15a127e71b4bb792ecb28a23b4ad90fe69654261dde"},
    {"layers.0.W11.weight", {kTensor8Shape, 2 }, "float32", "df035173eaec7ba5e56e2536e9666b03a774233b429e096164e74ca96dcef2b0"},
    {"layers.0.W12.bias", {kTensor9Shape, 1 }, "float32", "70e16dc4d3dc2a4b9fb1b01985db96ce90c98469514083c77e461bdb4b4b9acb"},
    {"layers.0.W12.weight", {kTensor10Shape, 2 }, "float32", "21aefb71b6c7826214d27cf2890aad26cf1072992541ca4593aa035197691e64"},
    {"layers.0.W13.bias", {kTensor11Shape, 1 }, "float32", "d52b9c9cebf9d6cb7fbdea2fc681d1f41ce4b003e5f5cb88d7a962b1e33f9ea1"},
    {"layers.0.W13.weight", {kTensor12Shape, 2 }, "float32", "9b0c0239d318b0e97935ba5550a9d0a2685fc325ffb840308ff692f430365ab0"},
    {"layers.0.W2.bias", {kTensor13Shape, 1 }, "float32", "c191a3bddc08f77dbc9a5c877a8cd203b4ddedeff0b26f20a85605ea0989a64e"},
    {"layers.0.W2.weight", {kTensor14Shape, 2 }, "float32", "7be1b7f16cb842a94d7d215c154f9ea317fd57792bb9018fd4e1ed0a049ba85d"},
    {"layers.0.W3.bias", {kTensor15Shape, 1 }, "float32", "819bcaed0277ff5319e4290bf9314fc9f63de199086e49e1337e209c2682c1f7"},
    {"layers.0.W3.weight", {kTensor16Shape, 2 }, "float32", "4e3c3f51eb04b3b1d7193be77f1d82ae31280edd080739a6ce56335a092921bc"},
    {"layers.0.ffn.W_in.bias", {kTensor17Shape, 1 }, "float32", "e1ca21a2dbfad2d653aa1c13c4f467c79dbe8c7747107972ce3072ba987851ec"},
    {"layers.0.ffn.W_in.weight", {kTensor18Shape, 2 }, "float32", "ef4048ed3ee5746b8d0fe48332ca0269717d6305c33dfc611982a95742b1065c"},
    {"layers.0.ffn.W_out.bias", {kTensor19Shape, 1 }, "float32", "7d9ba74ad1490901a333c6c849297c327777260609342f7a9b234e09185b13d0"},
    {"layers.0.ffn.W_out.weight", {kTensor20Shape, 2 }, "float32", "cf834dc08c39a9107f4c66a9484753a7fc05656ee7eb2eda35dc8b353461d099"},
    {"layers.0.norm1.bias", {kTensor21Shape, 1 }, "float32", "fec9aa94f3826cf8a2e4202242c3d576242dca2e01c01101fee576bc0f46abdc"},
    {"layers.0.norm1.weight", {kTensor22Shape, 1 }, "float32", "86e2effa0f0a61dddbe33becea7a8a83d54847ee0553e7b0b5c75b89726c7985"},
    {"layers.0.norm2.bias", {kTensor23Shape, 1 }, "float32", "248162c25f0572ae9a00a099e3b9da44e1d452ea5c1d9903d0050e2f0b38b58f"},
    {"layers.0.norm2.weight", {kTensor24Shape, 1 }, "float32", "bc1a215111d97e0fbf07c7a7d58836d2d8e49c9a6f19c42bf734737724efb6c6"},
    {"layers.0.norm3.bias", {kTensor25Shape, 1 }, "float32", "b015c40954a0c39c1bbfb3251ffe7fe441c1fb054a808b3e861c3415c06088ca"},
    {"layers.0.norm3.weight", {kTensor26Shape, 1 }, "float32", "f85f6b6b9c3a517fc33f036affc3025675e310e9b856336bea966482edaa24a2"},
    {"layers.1.W1.bias", {kTensor27Shape, 1 }, "float32", "472ca0f0938fec16b84a2282108c5ff3e844ad156157442abc67faac9a6fbc37"},
    {"layers.1.W1.weight", {kTensor28Shape, 2 }, "float32", "8f9b400988ce184a512d161c8823b50857c0e9bb37207ad3a315c3f1b2632d83"},
    {"layers.1.W11.bias", {kTensor29Shape, 1 }, "float32", "ab514e4afd9fd2ea49b94317b390f366f29b76ff873c1223ede94618c96fdd30"},
    {"layers.1.W11.weight", {kTensor30Shape, 2 }, "float32", "0f27cd1c983de1123dc7df2dce0045f26f8466c2137605b600ebe3ba0c926f43"},
    {"layers.1.W12.bias", {kTensor31Shape, 1 }, "float32", "d95292953ab02ac4db003652e75ea6ac7895a782511fb47c6cc422988dc6a7f4"},
    {"layers.1.W12.weight", {kTensor32Shape, 2 }, "float32", "23a1679bec832ec6d193fee6b20c56508d1e7d3f55e4b55277f3794b05c4d44a"},
    {"layers.1.W13.bias", {kTensor33Shape, 1 }, "float32", "e7dc7eb3a748b41c0a62d4b23f36b0edbfe463061e66710bbc9680e47bea1103"},
    {"layers.1.W13.weight", {kTensor34Shape, 2 }, "float32", "1d07bab5c2de320b2e673949094d9dd4385a926c1f9fb05403f116bf95a527e1"},
    {"layers.1.W2.bias", {kTensor35Shape, 1 }, "float32", "dc09d0fbec83e8cd6b8e028fedff5cac03e344882d23e8cb73e5a3b3d5fe19ec"},
    {"layers.1.W2.weight", {kTensor36Shape, 2 }, "float32", "5b2b310b3a6e5ea4583812e0faac947e14309b7e51fd88b3385c33c20f7658e8"},
    {"layers.1.W3.bias", {kTensor37Shape, 1 }, "float32", "a984d3ee2bc4918f2e886f9932405f0f077a8b7ff662cd07b2abe5b70fde2aa0"},
    {"layers.1.W3.weight", {kTensor38Shape, 2 }, "float32", "129b112544d1dc6b0608130801d9abcc327c592f3e3dccb26602d34f6bef94e5"},
    {"layers.1.ffn.W_in.bias", {kTensor39Shape, 1 }, "float32", "6b24b1731f9014cac29cf77248d767f7a61216128d032181a74e4da929adc796"},
    {"layers.1.ffn.W_in.weight", {kTensor40Shape, 2 }, "float32", "cb0cea671ef0f7b4d3fdfbb6379ba20d6e45fc3d81d414f7fb3d508ce56ca643"},
    {"layers.1.ffn.W_out.bias", {kTensor41Shape, 1 }, "float32", "3d70e76ca0146fa19138c0318d6a025cf247d78695c2cb9fc08f5c8ba1d4b1b1"},
    {"layers.1.ffn.W_out.weight", {kTensor42Shape, 2 }, "float32", "f059197c9c6e5c104e0a15237ba7773a56b48034802908cfc24e022ae57ce340"},
    {"layers.1.norm1.bias", {kTensor43Shape, 1 }, "float32", "a50a48d5b7da5befcee2c3aaa3ef4d0470f96e06e0807bd4b4b66c1e2b7c6a30"},
    {"layers.1.norm1.weight", {kTensor44Shape, 1 }, "float32", "e7c116c13c5fa281cad23256fdfb011802cf75507fd121b351128472f21329c2"},
    {"layers.1.norm2.bias", {kTensor45Shape, 1 }, "float32", "56e362b1f14fb8eea2463f93af1298cbff8553951e6246198b0ade1ecff97f19"},
    {"layers.1.norm2.weight", {kTensor46Shape, 1 }, "float32", "db11d9e5878ca66a024c0a18948bcae8c7b180d3e2fd1d932ee72023d25e13a8"},
    {"layers.1.norm3.bias", {kTensor47Shape, 1 }, "float32", "1e73e21d1d7e38ff5f88739e25f7839a9b30083e802c69702ddaa80e5ee75ab4"},
    {"layers.1.norm3.weight", {kTensor48Shape, 1 }, "float32", "259f3f85c055d6476bc0828d2d54a6e42c2654bd19e20a7a45ee1e79b8c43a1c"},
    {"layers.2.W1.bias", {kTensor49Shape, 1 }, "float32", "5272879cf9789b92afe2f110609b7d68ecf9e6630528230c662e9ea06a755f75"},
    {"layers.2.W1.weight", {kTensor50Shape, 2 }, "float32", "00a2b725f5854dee35bfb59f49adf53a30d0c9152ff36d0bcf6585bc78a19a93"},
    {"layers.2.W11.bias", {kTensor51Shape, 1 }, "float32", "983a7a80afe9782644dd7377aafe92d9d3d0eaea386cd552badaee1d7dd21c7e"},
    {"layers.2.W11.weight", {kTensor52Shape, 2 }, "float32", "46965b111180dc54f93fc17cf0a5701fad2aaeddaa18e43f3debb6362511ec0f"},
    {"layers.2.W12.bias", {kTensor53Shape, 1 }, "float32", "b4aa60dd2a3bf264f65d56e6327f7c0fc00db4e9134c1c71749fc91f4f22d3f5"},
    {"layers.2.W12.weight", {kTensor54Shape, 2 }, "float32", "d41dce9ddbbd899d64e0e904f579edadd4cf53aa01fea64db179b89448783821"},
    {"layers.2.W13.bias", {kTensor55Shape, 1 }, "float32", "1928c33be5794848d3aa00110c186312f16d57c3dfc8addb14698da2afc7cdec"},
    {"layers.2.W13.weight", {kTensor56Shape, 2 }, "float32", "b4d87cb02d821c0a98c4c178fbfdd962abde5d245680d77242217841e5eb4786"},
    {"layers.2.W2.bias", {kTensor57Shape, 1 }, "float32", "7e6862647a1bc9a1cb884a7efc6023d1a96ae4830af17abef5760b019be79bf9"},
    {"layers.2.W2.weight", {kTensor58Shape, 2 }, "float32", "b726f959c55fcd6900e127e1eabf42ce53bd19b0cce50f362e6d12412c8bb172"},
    {"layers.2.W3.bias", {kTensor59Shape, 1 }, "float32", "f20e040d75d2c09882d4c4db290c6383f4982fdbcf9d23d474854333b36389b3"},
    {"layers.2.W3.weight", {kTensor60Shape, 2 }, "float32", "cca3059fac45649a7814337a81c398fa41b9932a2018d9c866a4d09a7c56f021"},
    {"layers.2.ffn.W_in.bias", {kTensor61Shape, 1 }, "float32", "30d238a36838834165007b909b360137b09792ba50194edc480b54c1e073b5c1"},
    {"layers.2.ffn.W_in.weight", {kTensor62Shape, 2 }, "float32", "efee0364e8ca2bed7712d24c644119240f916a66754048d923e0e87081816cbd"},
    {"layers.2.ffn.W_out.bias", {kTensor63Shape, 1 }, "float32", "47d00741b2bdac581ac1ecaa51d5f223f22024a8dddebf1f9c014b255ef59c05"},
    {"layers.2.ffn.W_out.weight", {kTensor64Shape, 2 }, "float32", "d2ab4e83eb22595f9b503049cbf3836eef0a4670ec4aa2d5f94a8dfff953be14"},
    {"layers.2.norm1.bias", {kTensor65Shape, 1 }, "float32", "3f9b9fbb803d341153e3d8367148dd352f645d0ef82453fbf15de35c0a6691c6"},
    {"layers.2.norm1.weight", {kTensor66Shape, 1 }, "float32", "a0e09d9895acca3c41c018f608bbffedead4a9a47b7088b6a4ef367771580acc"},
    {"layers.2.norm2.bias", {kTensor67Shape, 1 }, "float32", "566b4949e84e8455ee3e59b902135c37949da098b053345a45b18340352a023b"},
    {"layers.2.norm2.weight", {kTensor68Shape, 1 }, "float32", "5a9f3673644212ff5d7b0a4443afd5b4c69d8846805739844646202512d0e655"},
    {"layers.2.norm3.bias", {kTensor69Shape, 1 }, "float32", "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1"},
    {"layers.2.norm3.weight", {kTensor70Shape, 1 }, "float32", "2f20cd03c9cd392a406c56232b0ff93a15f6d6d7da79086bfa14f55d4a4031b0"},
    {"positional_encoding.bias", {kTensor71Shape, 1 }, "float32", "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"},
    {"positional_encoding.weight", {kTensor72Shape, 2 }, "float32", "388280fda2cc00225fcc5ce2ea987f2a9117aa4a7652ba2432f53f961a5b5b9d"},
};

constexpr WeightManifestView kDefaultMpnnD64Manifest{
    "0.1.0",
    kDefaultMpnnD64ModelName,
    kDefaultMpnn64ModelFamily,
    "archive-embedded-header-0348104439a78dae",
    kDefaultMpnn64HiddenDim,
    kDefaultMpnn64NeighborCount,
    kDefaultMpnn64RbfCount,
    kDefaultMpnn64RbfFeatureOrder,
    kDefaultMpnn64LayerCount,
    "safetensors:hikoboshi-mpnn-d64-v1;runtime_tensors=73;excluded=gap,gap_open",
    kDefaultMpnn64MessageScale,
    "float32",
    {kTensorManifests, 73},
    "c61e079b539af5e31ba145ab91f7f607634295e90240e444a5bf34013304175b",
    "sha256",
    "archive-embedded-header-0348104439a78dae",
    "0348104439a78dae87012512408a203a4bf96755554df60fde3f21a8eafb1951",
    "extract_embedded_mpnn64_header (Megure Labs internal)",
    "i08-20260428",
    "2026-04-28",
    kHardSwGapFamily,
    kHardSwDefaultGapOpen,
    kHardSwDefaultGapExtension,
    kSoftSwGapFamily,
    kSoftSwMpnn64GapOpen,
    kSoftSwMpnn64GapExtension,
    "raw_dot_product",
    "checksum-verified; hard-SW defaults preserved; soft-SW descriptor metadata recorded",
    "archive embedded header sha256 verified",
};

namespace proteinmpnn_generated =
    ::hikoboshi::weights::generated::proteinmpnn_v48_eps020;

const std::array<TensorManifestView, proteinmpnn_generated::kRuntimeTensorCount>&
proteinmpnn_v48_eps020_tensor_manifests() noexcept {
  static const auto kEntries = []() {
    std::array<TensorManifestView, proteinmpnn_generated::kRuntimeTensorCount>
        entries{};
    for (std::size_t index = 0;
         index < proteinmpnn_generated::kRuntimeTensorCount; ++index) {
      const proteinmpnn_generated::TensorBlobInfo& info =
          proteinmpnn_generated::kRuntimeTensors[index];
      entries[index] = TensorManifestView{
          info.name,
          {info.shape, info.rank},
          info.dtype,
          info.checksum,
      };
    }
    return entries;
  }();
  return kEntries;
}

constexpr std::string_view kProteinMpnnV48020ModelVersion{"v_48_020"};
constexpr std::string_view kProteinMpnnV48020SourceCheckpoint{
    "external/ProteinMPNN/vanilla_model_weights/v_48_020.pt"};
constexpr std::string_view kProteinMpnnV48020GenerationTool{
    "convert_proteinmpnn_v48_020_to_safetensors.py (Megure Labs internal)"};
constexpr std::string_view kProteinMpnnV48020GenerationToolVersion{
    "mpnn-01-weights-v48-020-20260630"};
constexpr std::string_view kProteinMpnnV48020GenerationDate{"2026-06-30"};
constexpr std::string_view kProteinMpnnV48020ValidationStatus{
    "checksum-verified; per-tensor SHA-256 recorded; release_certified=true; "
    "mpnn-05 baked ProteinMPNN v_48_020 parity passed"};
constexpr std::string_view kProteinMpnnV48020ProvenanceStatus{
    "vanilla ProteinMPNN v_48_020 checkpoint converted to float32 "
    "safetensors; no tensor reordering inside individual tensors; "
    "validated against baked upstream v_48_020 parity goldens"};

const WeightManifestView& proteinmpnn_v48_eps020_manifest_storage() noexcept {
  static const WeightManifestView kManifest{
      std::string_view{"0.1.0"},
      kDefaultProteinMpnnV48Eps020ModelName,
      kDefaultProteinMpnnV48020ModelFamily,
      kProteinMpnnV48020ModelVersion,
      kDefaultProteinMpnnV48020HiddenDim,
      kDefaultProteinMpnnV48020NeighborCount,
      kDefaultProteinMpnnV48020RbfCount,
      kDefaultMpnn64RbfFeatureOrder,
      kDefaultProteinMpnnV48020LayerCount,
      proteinmpnn_generated::kTensorSchema,
      kDefaultProteinMpnnV48020MessageScale,
      std::string_view{"float32"},
      {proteinmpnn_v48_eps020_tensor_manifests().data(),
       proteinmpnn_v48_eps020_tensor_manifests().size()},
      proteinmpnn_generated::kSafetensorsBlobSha256,
      std::string_view{"sha256"},
      kProteinMpnnV48020SourceCheckpoint,
      proteinmpnn_generated::kSourceArtifactSha256,
      kProteinMpnnV48020GenerationTool,
      kProteinMpnnV48020GenerationToolVersion,
      kProteinMpnnV48020GenerationDate,
      kInverseFoldingGapFamily,
      0.0F,
      0.0F,
      kInverseFoldingGapFamily,
      0.0F,
      0.0F,
      kInverseFoldingSimilarity,
      kProteinMpnnV48020ValidationStatus,
      kProteinMpnnV48020ProvenanceStatus,
  };
  return kManifest;
}

}  // namespace

const WeightManifestView& default_mpnn_d64_manifest() noexcept {
  return kDefaultMpnnD64Manifest;
}

const WeightManifestView& default_mpnn64_manifest() noexcept {
  return default_mpnn_d64_manifest();
}

const WeightManifestView& default_proteinmpnn_v48_eps020_manifest() noexcept {
  return proteinmpnn_v48_eps020_manifest_storage();
}

const WeightManifestView& default_proteinmpnn_v48_020_manifest() noexcept {
  return default_proteinmpnn_v48_eps020_manifest();
}

}  // namespace hikoboshi::weights
