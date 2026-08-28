#ifndef HIKOBOSHI_DISPATCH_THREAD_POOL_HPP
#define HIKOBOSHI_DISPATCH_THREAD_POOL_HPP

/// @file
/// Compatibility alias for the private universal-detail thread pool.

#include <hikoboshi/universal/detail/thread_pool.hpp>

namespace hikoboshi::dispatch {

using ThreadPool = hikoboshi::universal::detail::ThreadPool;

}  // namespace hikoboshi::dispatch

#endif  // HIKOBOSHI_DISPATCH_THREAD_POOL_HPP
