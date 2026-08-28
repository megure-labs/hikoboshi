#include <hikoboshi/algorithms/metrics.hpp>

#include <cctype>
#include <cstddef>
#include <cstdint>

namespace hikoboshi::algorithms {
namespace {

bool has_codes(hikoboshi::universal::Span<const char> codes,
               std::size_t length) noexcept {
  return codes.data != nullptr && codes.size >= length;
}

bool is_standard_known_amino_acid(char code) noexcept {
  switch (static_cast<char>(std::toupper(static_cast<unsigned char>(code)))) {
    case 'A':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'P':
    case 'Q':
    case 'R':
    case 'S':
    case 'T':
    case 'V':
    case 'W':
    case 'Y':
      return true;
    default:
      return false;
  }
}

}  // namespace

hikoboshi::universal::MetricValue compute_identity(
    const hikoboshi::universal::AlignmentPath& path,
    hikoboshi::universal::Span<const char> query_codes,
    hikoboshi::universal::Span<const char> target_codes) noexcept {
  if (!has_codes(query_codes, 1) || !has_codes(target_codes, 1)) {
    return invalid_metric(
        hikoboshi::universal::MetricInvalidReason::MissingSequenceMetadata);
  }

  std::size_t denominator = 0;
  std::size_t matches = 0;
  for (const auto& step : path.steps) {
    if (step.query_index < 0 || step.target_index < 0) {
      continue;
    }
    const auto qi = static_cast<std::size_t>(step.query_index);
    const auto ti = static_cast<std::size_t>(step.target_index);
    if (qi >= query_codes.size || ti >= target_codes.size) {
      continue;
    }
    const char query_code = query_codes.data[qi];
    const char target_code = target_codes.data[ti];
    if (!is_standard_known_amino_acid(query_code) ||
        !is_standard_known_amino_acid(target_code)) {
      continue;
    }
    ++denominator;
    if (std::toupper(static_cast<unsigned char>(query_code)) ==
        std::toupper(static_cast<unsigned char>(target_code))) {
      ++matches;
    }
  }

  if (denominator == 0) {
    return invalid_metric(
        hikoboshi::universal::MetricInvalidReason::ZeroDenominator);
  }
  return valid_metric(static_cast<double>(matches) /
                      static_cast<double>(denominator));
}

}  // namespace hikoboshi::algorithms
