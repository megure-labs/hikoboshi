#include <exception>
#include <iostream>
#include <string_view>

namespace hikoboshi::cli {

bool is_help_flag(std::string_view arg) noexcept;
int exit_code_invalid_arguments() noexcept;
int exit_code_internal_failure() noexcept;

int run_encode(int argc, char** argv);
int run_pairwise(int argc, char** argv);
int run_pair_list(int argc, char** argv);
int run_all_vs_all(int argc, char** argv);
int run_score_alignment(int argc, char** argv);
int run_design(int argc, char** argv);
int run_info(int argc, char** argv);
int run_version(int argc, char** argv);

namespace {

void print_usage(std::ostream& out) {
  out << "usage: hikoboshi <command> [args]\n"
      << "\n"
      << "commands:\n"
      << "  encode <input>\n"
      << "  pairwise <query> <target>\n"
      << "  pair-list --pairs FILE.tsv (--fasta FILE.fa | PDB_DIR)\n"
      << "  all-vs-all <mode> <inputs...>\n"
      << "  score-alignment --query Q --target T --correspondences PAIRS.tsv\n"
      << "  design --pdb INPUT.pdb --out OUT.fasta\n"
      << "  info\n"
      << "  version\n";
}

}  // namespace

int run_main(int argc, char** argv) {
  if (argc <= 1) {
    print_usage(std::cerr);
    return exit_code_invalid_arguments();
  }

  const std::string_view command{argv[1]};
  if (is_help_flag(command)) {
    print_usage(std::cout);
    return 0;
  }
  if (command == "--version") {
    std::cout << "hikoboshi 0.1.0 placeholder\n";
    return 0;
  }
  if (command == "encode") {
    return run_encode(argc - 2, argv + 2);
  }
  if (command == "pairwise") {
    return run_pairwise(argc - 2, argv + 2);
  }
  if (command == "pair-list" || command == "pair_list") {
    return run_pair_list(argc - 2, argv + 2);
  }
  if (command == "all-vs-all" || command == "all_vs_all") {
    return run_all_vs_all(argc - 2, argv + 2);
  }
  if (command == "score-alignment" || command == "score_alignment") {
    return run_score_alignment(argc - 2, argv + 2);
  }
  if (command == "design" || command == "inverse-fold") {
    return run_design(argc - 2, argv + 2);
  }
  if (command == "info") {
    return run_info(argc - 2, argv + 2);
  }
  if (command == "version") {
    return run_version(argc - 2, argv + 2);
  }

  std::cerr << "hikoboshi: unknown command: " << command << '\n';
  print_usage(std::cerr);
  return exit_code_invalid_arguments();
}

}  // namespace hikoboshi::cli

int main(int argc, char** argv) {
  try {
    return hikoboshi::cli::run_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "hikoboshi: internal_error: " << error.what() << '\n';
    return hikoboshi::cli::exit_code_internal_failure();
  } catch (...) {
    std::cerr << "hikoboshi: internal_error\n";
    return hikoboshi::cli::exit_code_internal_failure();
  }
}
