#include <hikoboshi/io/all_vs_all_layout.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace pio = hikoboshi::io;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "all_vs_all_artifact_layout_tests: %s\n", message);
  std::exit(1);
}

void test_sanitized_identifier_rules() {
  if (pio::sanitize_identifier("A-Z.9_ok") != "A-Z.9_ok") {
    fail("allowed identifier characters must be preserved");
  }
  if (pio::sanitize_identifier("a  b///c__d@@e") != "a_b_c_d_e") {
    fail("disallowed characters must become collapsed underscores");
  }
}

void test_stable_ids_preserve_index_for_collisions() {
  const pio::ArtifactInputId first{0, "/tmp/query file.pdb", {}, "A", {}, 1};
  const pio::ArtifactInputId second{
      1, "/other/query file.cif", {}, "B/2", "2 alt", 1};

  if (pio::file_stem_from_path(first.source_path) != "query file") {
    fail("file stem extraction mismatch");
  }
  if (pio::stable_input_id(first) != "0001_query_file_A_model1" ||
      pio::stable_input_id(second) !=
          "0002_query_file_B_2_model2_alt") {
    fail("stable input IDs must sanitize stems and preserve index prefixes");
  }
}

void test_pair_paths_use_chartered_directory_shape() {
  const pio::ArtifactInputId first{0, "/tmp/query.pdb", {}, "A", {}, 1};
  const pio::ArtifactInputId second{1, "/tmp/target.pdb", {}, "B", {}, 1};
  const auto paths = pio::pair_artifact_paths("output", first, second);
  if (paths.status.code != hiko_u::StatusCode::Ok) {
    fail("pair path construction should succeed");
  }
  const std::string pair_id =
      "0001_query_A_model1__0002_target_B_model1";
  if (paths.value.pair_id != pair_id ||
      paths.value.fasta_path != "output/alignments/" + pair_id + ".fasta" ||
      paths.value.pdb_path != "output/pdb/" + pair_id + ".pdb") {
    fail("pair paths must use output/alignments and output/pdb layout");
  }
}

void test_all_vs_all_layout_order() {
  const std::vector<pio::ArtifactInputId> inputs = {
      {0, "a.pdb", {}, "A", {}, 1},
      {1, "b.pdb", {}, "A", {}, 1},
      {2, "c.pdb", {}, "A", {}, 1},
  };
  const auto paths = pio::build_all_vs_all_artifact_layout(
      "out", {inputs.data(), inputs.size()}, false);
  if (paths.status.code != hiko_u::StatusCode::Ok || paths.value.size() != 3) {
    fail("default all-vs-all layout should contain N(N-1)/2 paths");
  }
  const std::size_t expected[][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (std::size_t i = 0; i < 3; ++i) {
    if (paths.value[i].query_index != expected[i][0] ||
        paths.value[i].target_index != expected[i][1]) {
      fail("all-vs-all artifact paths must be in stable lexicographic order");
    }
  }

  const auto with_self = pio::build_all_vs_all_artifact_layout(
      "out", {inputs.data(), inputs.size()}, true);
  if (with_self.status.code != hiko_u::StatusCode::Ok ||
      with_self.value.size() != 6 ||
      with_self.value[0].query_index != 0 ||
      with_self.value[0].target_index != 0 ||
      with_self.value[5].query_index != 2 ||
      with_self.value[5].target_index != 2) {
    fail("include-self layout order mismatch");
  }
}

}  // namespace

int main() {
  test_sanitized_identifier_rules();
  test_stable_ids_preserve_index_for_collisions();
  test_pair_paths_use_chartered_directory_shape();
  test_all_vs_all_layout_order();
  return 0;
}
