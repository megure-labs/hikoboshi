#include <hikoboshi/dispatch/cpu_features.hpp>
#include <hikoboshi/dispatch/dispatch_table.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace hiko_d = hikoboshi::dispatch;
namespace hiko_p = hikoboshi::primitives;
namespace hiko_u = hikoboshi::universal;

namespace {

bool nearly_equal(float lhs, float rhs) {
  return std::fabs(lhs - rhs) <= 1e-4F;
}

void fail(const char* message) {
  std::fprintf(stderr, "dispatch_table_smoke: %s\n", message);
  std::exit(1);
}

void test_table_slots_are_present() {
  const hiko_d::DispatchTable& table = hiko_d::scalar_dispatch_table();
  if (table.backend != hiko_u::Backend::Scalar || table.knn == nullptr ||
      table.rbf == nullptr || table.gather == nullptr ||
      table.layer_norm == nullptr || table.reduce_sum_rows == nullptr ||
      table.reduce_mean_rows == nullptr || table.softmax == nullptr ||
      table.log_softmax == nullptr || table.gemm_nn == nullptr ||
      table.gemm_nt == nullptr || table.smith_waterman == nullptr ||
      table.traceback == nullptr) {
    fail("scalar dispatch table must expose every primitive slot");
  }
}

void test_selected_table_is_scalar() {
  const hiko_d::CpuFeatures& features = hiko_d::detected_cpu_features();
  (void)features;

  const hiko_d::DispatchTable& scalar = hiko_d::scalar_dispatch_table();
  const hiko_d::DispatchTable& selected = hiko_d::selected_dispatch_table();
  if (&selected != &scalar || selected.backend != hiko_u::Backend::Scalar) {
    fail("Hikoboshi 0.1.0 selected dispatch table must stay scalar-only");
  }
}

void test_reserved_backends_do_not_select_scalar() {
  if (!hiko_d::selects_scalar(hiko_u::Backend::Auto) ||
      !hiko_d::selects_scalar(hiko_u::Backend::Scalar)) {
    fail("Auto and Scalar must remain the only scalar selectors");
  }

  const hiko_u::Backend reserved_backends[] = {
      hiko_u::Backend::Sse4,   hiko_u::Backend::Avx2,  hiko_u::Backend::Avx512,
      hiko_u::Backend::Neon,   hiko_u::Backend::Sve,   hiko_u::Backend::Cuda,
      hiko_u::Backend::Hip,    hiko_u::Backend::Metal, hiko_u::Backend::Vulkan,
      hiko_u::Backend::OpenCl,
  };
  for (const hiko_u::Backend backend : reserved_backends) {
    if (hiko_d::selects_scalar(backend)) {
      fail("reserved backend enum values must not be working scalar modes");
    }
  }
}

void test_table_gemm_nt_matches_scalar_result() {
  const float lhs[2 * 3] = {
      1.0F, 2.0F, 3.0F,
      4.0F, 5.0F, 6.0F,
  };
  const float rhs[2 * 3] = {
      7.0F, 8.0F, 9.0F,
      10.0F, 11.0F, 12.0F,
  };
  float output[2 * 2] = {};
  hiko_p::linalg::GemmScalarRequest request{};
  request.lhs = lhs;
  request.rhs = rhs;
  request.m = 2;
  request.n = 2;
  request.k = 3;

  hiko_d::selected_dispatch_table().gemm_nt(request, output);

  if (!nearly_equal(output[0], 50.0F) || !nearly_equal(output[1], 68.0F) ||
      !nearly_equal(output[2], 122.0F) || !nearly_equal(output[3], 167.0F)) {
    fail("dispatch table gemm_nt slot must preserve scalar results");
  }
}

}  // namespace

int main() {
  test_table_slots_are_present();
  test_selected_table_is_scalar();
  test_reserved_backends_do_not_select_scalar();
  test_table_gemm_nt_matches_scalar_result();
  return 0;
}
