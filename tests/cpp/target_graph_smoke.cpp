#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/types.hpp>

#include <type_traits>

namespace hiko = hikoboshi::api;

static_assert(
    std::is_same<decltype(&hiko::PairwiseResultSink::receive),
                 hikoboshi::universal::Status (hiko::PairwiseResultSink::*)(
                     const hiko::PairwiseResultRecord&)>::value,
    "public all-vs-all sink must remain on the API surface");

int main() {
  return 0;
}
