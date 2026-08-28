#ifndef HIKOBOSHI_UNIVERSAL_VERSION_HPP
#define HIKOBOSHI_UNIVERSAL_VERSION_HPP

/// @file
/// Hikoboshi semantic version constants.

#include <string_view>

namespace hikoboshi::universal {

/// Product semantic version view.
///
/// `label` is empty for a normal release and may carry a prerelease or build
/// label in development builds.
struct VersionView {
  int major;
  int minor;
  int patch;
  std::string_view label;
};

/// Hikoboshi public API version constants for the 0.1.0 release line.
inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;
inline constexpr std::string_view kVersionLabel{};
inline constexpr VersionView kVersion{
    kVersionMajor,
    kVersionMinor,
    kVersionPatch,
    kVersionLabel,
};

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_VERSION_HPP
