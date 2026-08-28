#include <hikoboshi/io/fasta_writer.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace hiko = hikoboshi::api;
namespace pio = hikoboshi::io;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "fasta_writer_tests: %s\n", message);
  std::exit(1);
}

hiko::AlignmentPath alignment_path() {
  hiko::AlignmentPath path{};
  path.steps.push_back({0, 0, 1.0F});
  path.steps.push_back({1, hiko_u::kAlignmentGapSentinel, 1.0F});
  path.steps.push_back({2, 1, 1.0F});
  path.steps.push_back({hiko_u::kAlignmentGapSentinel, 2, 1.0F});
  path.steps.push_back({3, 3, 1.0F});
  path.aligned_pairs = 3;
  path.query_start = 0;
  path.query_end = 3;
  path.target_start = 0;
  path.target_end = 3;
  return path;
}

pio::FastaInputMetadata metadata(const char* id,
                                 const std::vector<char>& codes) {
  return {id, {codes.data(), codes.size()}, codes.size(), true};
}

void test_gapped_fasta_shape_and_wrapping() {
  const std::vector<char> query = {'A', 'C', 'D', 'E'};
  const std::vector<char> target = {'A', 'W', 'Y', 'E'};
  const pio::FastaWriterOptions options{4};

  const auto rendered = pio::render_alignment_fasta(
      alignment_path(), metadata("query alpha", query),
      metadata("target/beta", target), options);
  if (rendered.status.code != hiko_u::StatusCode::Ok) {
    fail("FASTA render should succeed with sequence metadata");
  }

  const std::string expected =
      ">query_alpha\n"
      "ACD-\n"
      "E\n"
      ">target_beta\n"
      "A-WY\n"
      "E\n";
  if (rendered.value != expected) {
    fail("FASTA shape, gap conversion, or stable wrapping mismatch");
  }
}

void test_embedding_only_without_metadata_is_unavailable() {
  const pio::FastaInputMetadata embedding_only{
      "embedding.npy", {nullptr, 0}, 4, false};
  const std::vector<char> target = {'A', 'W', 'Y', 'E'};
  const auto rendered = pio::render_alignment_fasta(
      alignment_path(), embedding_only, metadata("target", target));
  if (rendered.status.code != hiko_u::StatusCode::Unavailable) {
    fail("embedding-only FASTA without sequence metadata must be unavailable");
  }
}

void test_embedding_with_sequence_metadata_is_allowed() {
  hiko::AlignmentPath path{};
  path.steps.push_back({0, 0, 1.0F});
  path.aligned_pairs = 1;
  const std::vector<char> query = {'M'};
  const std::vector<char> target = {'M'};

  const auto rendered = pio::render_alignment_fasta(
      path, metadata("query.embedding.npy", query),
      metadata("target.embedding.npy", target));
  if (rendered.status.code != hiko_u::StatusCode::Ok ||
      rendered.value.find(">query.embedding.npy\nM\n") == std::string::npos) {
    fail("embedding metadata should enable FASTA output");
  }
}

void test_invalid_path_index_is_not_silently_substituted() {
  hiko::AlignmentPath path{};
  path.steps.push_back({7, 0, 1.0F});
  const std::vector<char> query = {'A'};
  const std::vector<char> target = {'A'};

  const auto rendered =
      pio::render_alignment_fasta(path, metadata("query", query),
                                  metadata("target", target));
  if (rendered.status.code != hiko_u::StatusCode::InvalidArgument) {
    fail("invalid path indices must surface as invalid arguments");
  }
}

}  // namespace

int main() {
  test_gapped_fasta_shape_and_wrapping();
  test_embedding_only_without_metadata_is_unavailable();
  test_embedding_with_sequence_metadata_is_allowed();
  test_invalid_path_index_is_not_silently_substituted();
  return 0;
}
