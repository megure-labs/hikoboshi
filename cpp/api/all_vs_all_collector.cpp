#include <hikoboshi/api/all_vs_all.hpp>

#include <hikoboshi/api/engine.hpp>

namespace hikoboshi::api {

universal::Result<AllVsAllResult> collect_all_vs_all(
    const Engine& engine,
    const AllVsAllStructureRequest& request) {
  return engine.collect_all_vs_all(request);
}

universal::Result<AllVsAllResult> collect_all_vs_all(
    const Engine& engine,
    const AllVsAllCoordsRequest& request) {
  return engine.collect_all_vs_all(request);
}

universal::Result<AllVsAllResult> collect_all_vs_all(
    const Engine& engine,
    const AllVsAllEmbeddingRequest& request) {
  return engine.collect_all_vs_all(request);
}

}  // namespace hikoboshi::api
