#include <hikoboshi/api/version.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <iostream>
#include <string_view>

namespace hikoboshi::cli {

bool is_help_flag(std::string_view arg) noexcept;
int report_status(hikoboshi::universal::Status status);
hikoboshi::universal::Result<hikoboshi::universal::WeightsHandle>
resolve_default_weights() noexcept;
void render_info_backend_summary(
    std::ostream& out,
    const hikoboshi::api::BackendCapabilities& capabilities);
void render_info_backends(
    std::ostream& out,
    const hikoboshi::api::BackendCapabilities& capabilities);
void render_info_models(std::ostream& out);

namespace {

void print_info_usage(std::ostream& out) {
  out << "usage: hikoboshi info [backends|models]\n";
}

}  // namespace

int run_info(int argc, char** argv) {
  if (argc > 0 && is_help_flag(argv[0])) {
    print_info_usage(std::cout);
    return 0;
  }

  if (argc == 1) {
    const std::string_view topic{argv[0]};
    if (topic == "backends") {
      render_info_backends(std::cout, hikoboshi::api::backend_capabilities());
      return 0;
    }
    if (topic == "models") {
      render_info_models(std::cout);
      return 0;
    }
    print_info_usage(std::cerr);
    return report_status({hikoboshi::universal::StatusCode::InvalidArgument,
                          "unknown info topic"});
  }
  if (argc != 0) {
    print_info_usage(std::cerr);
    return report_status({hikoboshi::universal::StatusCode::InvalidArgument,
                          "info accepts at most one topic"});
  }

  const hikoboshi::api::VersionInfo version = hikoboshi::api::version_info();
  const hikoboshi::api::BackendCapabilities capabilities =
      hikoboshi::api::backend_capabilities();
  const hikoboshi::weights::WeightManifestView& manifest =
      hikoboshi::weights::default_mpnn_d64_manifest();
  const hikoboshi::universal::Result<hikoboshi::universal::WeightsHandle> weights =
      resolve_default_weights();
  if (weights.status.code != hikoboshi::universal::StatusCode::Ok) {
    return report_status(weights.status);
  }

  std::cout << "product\t" << version.product_name << '\n';
  std::cout << "version\t" << version.version.major << '.'
            << version.version.minor << '.' << version.version.patch << '\n';
  render_info_backend_summary(std::cout, capabilities);
  std::cout << "default_weights\t" << manifest.model_name << '\n';
  std::cout << "default_weights_status\tresolved\n";
  std::cout << "default_weights_hidden_dim\t" << manifest.hidden_dimension
            << '\n';
  std::cout << "default_weights_neighbor_count\t" << manifest.neighbor_count
            << '\n';
  std::cout << "default_weights_rbf_count\t" << manifest.rbf_count << '\n';
    std::cout << "default_weights_layer_count\t" << manifest.layer_count << '\n';
    std::cout << "gap_open\t" << manifest.gap_open << '\n';
    std::cout << "gap_extension\t" << manifest.gap_extension << '\n';
    std::cout << "soft_gap_open\t" << manifest.soft_gap_open << '\n';
    std::cout << "soft_gap_extension\t" << manifest.soft_gap_extension << '\n';
    std::cout << "similarity\t" << manifest.similarity << '\n';
  std::cout << "checksum\t" << manifest.checksum << '\n';
  return 0;
}

}  // namespace hikoboshi::cli
