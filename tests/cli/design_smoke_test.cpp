#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char* kPdb =
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1       1.500   1.500   0.000  1.00  0.00           C\n"
    "ATOM      6  N   ALA A   2       4.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      7  CA  ALA A   2       5.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      8  C   ALA A   2       6.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      9  O   ALA A   2       7.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM     10  CB  ALA A   2       5.500   1.500   0.000  1.00  0.00           C\n"
    "ATOM     11  N   ALA A   3       8.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM     12  CA  ALA A   3       9.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM     13  C   ALA A   3      10.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM     14  O   ALA A   3      11.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM     15  CB  ALA A   3       9.500   1.500   0.000  1.00  0.00           C\n"
    "END\n";

std::string shell_quote(const std::filesystem::path& path) {
  std::string text = path.string();
  std::string quoted = "'";
  for (char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted += "'";
  return quoted;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

bool valid_aa_sequence(const std::string& sequence) {
  if (sequence.size() != 3U) {
    return false;
  }
  for (char c : sequence) {
    if (std::string{"ACDEFGHIKLMNPQRSTVWYX"}.find(c) ==
        std::string::npos) {
      return false;
    }
  }
  return true;
}

std::vector<std::string> fasta_sequences(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::vector<std::string> sequences;
  std::string line;
  std::string current;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    if (line.front() == '>') {
      if (!current.empty()) {
        sequences.push_back(current);
        current.clear();
      }
      continue;
    }
    current += line;
  }
  if (!current.empty()) {
    sequences.push_back(current);
  }
  return sequences;
}

bool npz_shape_is_2_3_21(const std::filesystem::path& path) {
  const std::string bytes = read_file(path);
  if (bytes.size() < 64U) {
    return false;
  }
  const unsigned char* data =
      reinterpret_cast<const unsigned char*>(bytes.data());
  if (data[0] != 0x50 || data[1] != 0x4B || data[2] != 0x03 ||
      data[3] != 0x04) {
    return false;
  }
  const std::size_t name_len =
      static_cast<std::size_t>(data[26]) |
      (static_cast<std::size_t>(data[27]) << 8U);
  const std::size_t extra_len =
      static_cast<std::size_t>(data[28]) |
      (static_cast<std::size_t>(data[29]) << 8U);
  const std::size_t payload_offset = 30U + name_len + extra_len;
  if (payload_offset + 10U >= bytes.size()) {
    return false;
  }
  if (bytes.compare(payload_offset, 6U, "\x93NUMPY", 6U) != 0) {
    return false;
  }
  const std::size_t header_len =
      static_cast<std::size_t>(data[payload_offset + 8U]) |
      (static_cast<std::size_t>(data[payload_offset + 9U]) << 8U);
  const std::size_t header_offset = payload_offset + 10U;
  if (header_offset + header_len > bytes.size()) {
    return false;
  }
  const std::string header =
      bytes.substr(header_offset, header_len);
  return header.find("'shape': (2, 3, 21)") != std::string::npos;
}

int run_command(const std::string& command) {
  const int code = std::system(command.c_str());
  if (code != 0) {
    std::cerr << "command failed: " << command << "\n";
  }
  return code;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: design_smoke_test <hikoboshi-binary>\n";
    return 2;
  }
  const std::filesystem::path binary = argv[1];
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("hikoboshi_design_smoke_" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::create_directories(root);
  const std::filesystem::path pdb = root / "input.pdb";
  const std::filesystem::path out1 = root / "design1.fasta";
  const std::filesystem::path out2 = root / "design2.fasta";
  const std::filesystem::path logprobs = root / "lp.npz";
  {
    std::ofstream pdb_out(pdb);
    pdb_out << kPdb;
  }

  const std::string base_args =
      " --pdb " + shell_quote(pdb) + " --num-seqs 2 --sampling-temp 0.1"
      " --seed 0";
  if (run_command(shell_quote(binary) + " design" + base_args + " --out " +
                  shell_quote(out1)) != 0) {
    return 1;
  }
  if (std::filesystem::exists(logprobs)) {
    std::cerr << "log-prob artifact must be opt-in\n";
    return 1;
  }
  if (run_command(shell_quote(binary) + " inverse-fold" + base_args + " --out " +
                  shell_quote(out2) + " --out-logprobs " +
                  shell_quote(logprobs)) != 0) {
    return 1;
  }

  const std::vector<std::string> sequences = fasta_sequences(out1);
  if (sequences.size() != 2U || !valid_aa_sequence(sequences[0]) ||
      !valid_aa_sequence(sequences[1])) {
    std::cerr << "design FASTA must contain two length-3 AA sequences\n";
    return 1;
  }
  if (read_file(out1) != read_file(out2)) {
    std::cerr << "design output must be deterministic with the same seed\n";
    return 1;
  }
  if (!npz_shape_is_2_3_21(logprobs)) {
    std::cerr << "log-prob NPZ must contain shape (2, 3, 21)\n";
    return 1;
  }

  std::filesystem::remove_all(root);
  return 0;
}
