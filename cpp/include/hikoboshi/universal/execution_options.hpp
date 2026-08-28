#ifndef HIKOBOSHI_UNIVERSAL_EXECUTION_OPTIONS_HPP
#define HIKOBOSHI_UNIVERSAL_EXECUTION_OPTIONS_HPP

/// @file
/// Public execution preferences shared by API and adapter layers.

#include <cstdint>

#include <hikoboshi/universal/backend.hpp>

namespace hikoboshi::universal {

/// Execution preferences supplied to public API engines.
///
/// `backend` selects the requested backend. `thread_count` is the public
/// concurrency selector: zero means automatic, one forces serial execution,
/// and values greater than one request that many worker slots.
struct ExecutionOptions {
  Backend backend = Backend::Auto;
  std::uint32_t thread_count = 0;
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_EXECUTION_OPTIONS_HPP
