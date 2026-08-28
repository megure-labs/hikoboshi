#include <hikoboshi/api/engine.hpp>
#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

namespace hiko = hikoboshi::api;
namespace hiko_m = hikoboshi::modules;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

static std::size_t g_mpnn64_forward_calls = 0;

extern "C" hiko_u::Status
__real__ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE(
    const hiko_m::Mpnn64ForwardRequest& request,
    const hiko_m::Mpnn64ForwardOutput& output);

extern "C" hiko_u::Status
__wrap__ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE(
    const hiko_m::Mpnn64ForwardRequest& request,
    const hiko_m::Mpnn64ForwardOutput& output) {
  ++g_mpnn64_forward_calls;
  return
      __real__ZN9hikoboshi7modules21mpnn64_forward_scalarERKNS0_20Mpnn64ForwardRequestERKNS0_19Mpnn64ForwardOutputE(
          request, output);
}

namespace {

constexpr std::uint64_t kExpectedTinyStructureChecksum =
    0x3038b6aa9c8732dfull;
constexpr std::size_t kResidues = 3;
constexpr std::size_t kHidden = 64;

void fail(const char* message) {
  std::fprintf(stderr, "encode_direct_route_parity_tests: %s\n", message);
  std::exit(1);
}

void reset_forward_count() noexcept {
  g_mpnn64_forward_calls = 0;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t fnv1a_floats(const std::vector<float>& values) {
  std::uint64_t hash = 1469598103934665603ull;
  for (float value : values) {
    const std::uint32_t bits = float_bits(value);
    for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
      hash ^= static_cast<std::uint8_t>((bits >> (8U * byte)) & 0xffU);
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

struct StructureFixture {
  std::vector<float> coordinates;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<char> residue_codes;
  std::vector<hiko_u::ResidueMetadataView> residues;

  hiko_u::StructureView structure_view() const {
    return {residue_codes.size(),
            {coordinates.data(), coordinates.size()},
            {atom_sources.data(), atom_sources.size()},
            {residue_codes.data(), residue_codes.size()},
            {residues.data(), residues.size()},
            "encode-direct-fixture",
            "encode-direct-fixture.pdb",
            {}};
  }

  hiko::CoordsInputView coords_view() const {
    return {residue_codes.size(),
            {coordinates.data(), coordinates.size()},
            {atom_sources.data(), atom_sources.size()},
            {residue_codes.data(), residue_codes.size()},
            {residues.data(), residues.size()}};
  }
};

StructureFixture make_fixture() {
  StructureFixture fixture{};
  fixture.coordinates = {
      0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F,
      2.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.5F, 1.0F, 0.0F, 1.5F, 2.0F, 0.0F, 1.5F,
      2.0F, 1.0F, 1.5F, 1.0F, 1.0F, 1.5F,
      0.0F, 0.0F, 3.0F, 1.0F, 0.0F, 3.0F, 2.0F, 0.0F, 3.0F,
      2.0F, 1.0F, 3.0F, 1.0F, 1.0F, 3.0F,
  };
  fixture.atom_sources.assign(kResidues * hiko_u::kCanonicalAtomCount,
                              hiko_u::AtomSource::Observed);
  fixture.residue_codes = {'A', 'C', 'D'};
  fixture.residues = {
      {'A', std::string_view{"ALA"}, std::string_view{"A"},
       std::string_view{"1"}, 1, 101, '\0', std::string_view{"fixture"}, 0,
       std::string_view{"encode-direct-fixture.pdb"}, 10},
      {'C', std::string_view{"CYS"}, std::string_view{"A"},
       std::string_view{"1"}, 1, 102, '\0', std::string_view{"fixture"}, 1,
       std::string_view{"encode-direct-fixture.pdb"}, 20},
      {'D', std::string_view{"ASP"}, std::string_view{"A"},
       std::string_view{"1"}, 1, 103, '\0', std::string_view{"fixture"}, 2,
       std::string_view{"encode-direct-fixture.pdb"}, 30},
  };
  return fixture;
}

hiko::Engine make_engine() {
  const auto package_result = hiko_w::default_mpnn_d64_package();
  if (package_result.status.code != hiko_u::StatusCode::Ok ||
      package_result.value.descriptor == nullptr) {
    fail("default Hikoboshi-MPNN-64 package must be available");
  }
  hiko::EngineConfig config{};
  config.package = package_result.value;
  config.execution.backend = hiko_u::Backend::Scalar;
  return hiko::Engine(config);
}

void require_encoded_contract(const hiko::EncodeResult& result,
                              const StructureFixture& fixture) {
  if (result.embedding.residue_count != kResidues ||
      result.embedding.dimension != kHidden ||
      result.embedding.values.size() != kResidues * kHidden) {
    fail("encode result shape changed");
  }
  if (result.embedding.residue_codes != fixture.residue_codes) {
    fail("encode result residue codes changed");
  }
  if (result.embedding.residues.size() != fixture.residues.size() ||
      result.embedding.residues[1].residue_code != 'C' ||
      result.embedding.residues[1].residue_number != 102 ||
      result.embedding.residues[1].source_record_index != 20) {
    fail("encode result residue metadata changed");
  }
  const std::uint64_t hash = fnv1a_floats(result.embedding.values);
  if (hash != kExpectedTinyStructureChecksum) {
    std::fprintf(stderr,
                 "encode_direct_route_parity_tests: embedding checksum "
                 "drifted: expected 0x%016llx got 0x%016llx\n",
                 static_cast<unsigned long long>(
                     kExpectedTinyStructureChecksum),
                 static_cast<unsigned long long>(hash));
    std::exit(1);
  }
}

void test_structure_encode_is_direct(const hiko::Engine& engine,
                                     const StructureFixture& fixture) {
  reset_forward_count();
  const auto result =
      engine.encode(hiko::EncodeStructureRequest{fixture.structure_view()});
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("structure encode must return Ok");
  }
  if (g_mpnn64_forward_calls != 1U) {
    fail("structure encode must call MPNN-64 forward exactly once");
  }
  require_encoded_contract(result.value, fixture);
}

void test_coords_encode_is_direct(const hiko::Engine& engine,
                                  const StructureFixture& fixture) {
  reset_forward_count();
  const auto result =
      engine.encode(hiko::EncodeCoordsRequest{fixture.coords_view()});
  if (result.status.code != hiko_u::StatusCode::Ok) {
    fail("coords encode must return Ok");
  }
  if (g_mpnn64_forward_calls != 1U) {
    fail("coords encode must call MPNN-64 forward exactly once");
  }
  require_encoded_contract(result.value, fixture);
}

}  // namespace

int main() {
  // Strict parity mode is the bit-stable contract used by the golden hash.
  setenv("HIKOBOSHI_GEMM_PARITY_MODE", "strict", 1);
  const StructureFixture fixture = make_fixture();
  const hiko::Engine engine = make_engine();
  test_structure_encode_is_direct(engine, fixture);
  test_coords_encode_is_direct(engine, fixture);
  return 0;
}
