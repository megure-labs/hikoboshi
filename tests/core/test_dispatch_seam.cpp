#include <cpp/core/dispatch/dispatch_table.hpp>
#include <cpp/core/universal/memory/growable_arena.hpp>
#include <cpp/core/types/span.hpp>

#include <cstddef>
#include <cstdlib>

namespace {

struct Request {
  int lhs;
  int rhs;
};

struct Output {
  int sum;
};

void test_add_forward(void* request, void* output) noexcept {
  const Request* typed_request = static_cast<const Request*>(request);
  Output* typed_output = static_cast<Output*>(output);
  typed_output->sum = typed_request->lhs + typed_request->rhs;
}

[[noreturn]] void fail() {
  std::exit(1);
}

}  // namespace

int main() {
  namespace dispatch = hikoboshi::core::dispatch;
  namespace memory = hikoboshi::core::universal::memory;

  constexpr const char* kOpId = "hikoboshi.core.test.add.v1";

  if (!dispatch::register_scalar_op(kOpId, &test_add_forward)) {
    fail();
  }

  const dispatch::DispatchTable table = dispatch::selected_dispatch_table();
  if (!dispatch::selects_scalar(table.backend)) {
    fail();
  }
  if (table.resolve(kOpId) != &test_add_forward) {
    fail();
  }

  Request request{7, 35};
  Output output{0};
  if (!table.call(kOpId, &request, &output)) {
    fail();
  }
  if (output.sum != 42) {
    fail();
  }

  memory::GrowableArena arena{128};
  int* values = arena.allocate<int>(4);
  if (values == nullptr) {
    fail();
  }
  values[0] = 1;
  values[1] = 2;
  values[2] = 3;
  values[3] = 4;
  hikoboshi::core::Span<int> span{values, 4};
  if (span.size != 4 || span[0] != 1 || span[3] != 4) {
    fail();
  }
  if (arena.bytes_used() == 0 || arena.block_count() == 0) {
    fail();
  }

  arena.reset();
  if (arena.bytes_used() != 0) {
    fail();
  }

  return 0;
}
