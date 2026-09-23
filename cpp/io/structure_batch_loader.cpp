#include <hikoboshi/io/structure_loader.hpp>
#include <hikoboshi/universal/detail/thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <new>
#include <system_error>
#include <thread>
#include <utility>

namespace hikoboshi::io {

universal::Status load_structures_from_files(
    universal::Span<const std::string> paths,
    std::vector<LoadedStructure>& output,
    std::size_t thread_count,
    const StructureLoadOptions& options) {
  if (paths.size != 0 && paths.data == nullptr)
    return universal::invalid_argument_status("structure file list is invalid");
  try {
    const std::size_t requested = thread_count == 0
        ? std::max(1U, std::thread::hardware_concurrency()) : thread_count;
    const std::size_t workers = paths.size < 8 ? 1 :
        std::min({requested, paths.size, std::size_t{32}});
    std::vector<LoadedStructure> loaded;
    loaded.reserve(paths.size);
    if (workers == 1) {
      for (std::size_t index = 0; index < paths.size; ++index) {
        auto item = load_structure_from_file(paths.data[index], options);
        if (!item.status.ok()) return item.status;
        loaded.push_back(std::move(item.value));
      }
    } else {
      // Empty optionals avoid constructing a parser/metadata owner per file
      // on the caller thread. Each worker constructs only its active input.
      std::vector<std::optional<LoadedStructure>> slots(paths.size);
      struct Failure {
        std::size_t index;
        universal::Status status;
      };
      std::vector<Failure> failures(workers, {paths.size, universal::ok_status()});
      std::atomic<std::size_t> next{0}, first_failure{paths.size};
      universal::detail::ThreadPool pool(workers);
      pool.parallel_for(0, workers, [&](std::size_t worker, std::size_t, std::size_t) {
        for (;;) {
          const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
          if (index >= paths.size || index >= first_failure.load(std::memory_order_relaxed))
            break;
          universal::Status status = universal::ok_status();
          try {
            auto item = load_structure_from_file(paths.data[index], options);
            status = item.status;
            if (status.ok()) slots[index].emplace(std::move(item.value));
          } catch (const std::bad_alloc&) {
            status = universal::unavailable_status("structure loading allocation failed");
          }
          if (!status.ok()) {
            failures[worker] = {index, status};
            auto prior = first_failure.load(std::memory_order_relaxed);
            while (index < prior && !first_failure.compare_exchange_weak(
                prior, index, std::memory_order_relaxed)) {}
            break;
          }
        }
      });
      // Claiming is monotonic. Every input before the earliest failure was
      // claimed and completed before this join, even if a later error won
      // the race to stop further claims. Later speculative reads are bounded
      // by worker count and never determine the returned error.
      const auto failure = std::min_element(failures.begin(), failures.end(),
          [](const Failure& a, const Failure& b) { return a.index < b.index; });
      if (failure->index != paths.size) return failure->status;
      for (auto& item : slots) loaded.push_back(std::move(*item));
    }
    output = std::move(loaded);
    return universal::ok_status();
  } catch (const std::bad_alloc&) {
    return universal::unavailable_status("structure loading allocation failed");
  } catch (const std::system_error&) {
    return universal::unavailable_status("structure loading worker creation failed");
  }
}

}  // namespace hikoboshi::io
