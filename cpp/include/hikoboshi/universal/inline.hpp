#ifndef HIKOBOSHI_UNIVERSAL_INLINE_HPP
#define HIKOBOSHI_UNIVERSAL_INLINE_HPP

#if defined(__GNUC__) || defined(__clang__)
#  define HIKOBOSHI_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#  define HIKOBOSHI_FORCE_INLINE __forceinline
#else
#  define HIKOBOSHI_FORCE_INLINE inline
#endif

#endif  // HIKOBOSHI_UNIVERSAL_INLINE_HPP
