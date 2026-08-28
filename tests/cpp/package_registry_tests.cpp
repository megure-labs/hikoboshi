#include <hikoboshi/weights/provider.hpp>
#include <hikoboshi/weights/provider_register.hpp>

#include <hikoboshi/universal/version.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace hiko_w = hikoboshi::weights;
namespace hiko_u = hikoboshi::universal;

namespace {

void fail(const char* message) {
  std::fprintf(stderr, "package_registry_tests: %s\n", message);
  std::exit(1);
}

constexpr std::size_t kExpectedCompiledPackages010 = 3;
constexpr std::string_view kExpectedAliases010[] = {
    "mpnn64",
    "mpnn-64",
    "Hikoboshi-MPNN-64",
};
constexpr std::string_view kExpectedEsm2_8mAliases010[] = {
    "esm2-8m",
    "esm2_8m",
    "Hikoboshi-ESM2-8M",
};
constexpr std::string_view kExpectedProteinMpnnV48020Aliases010[] = {
    "v_48_020",
    "Hikoboshi-ProteinMPNN-v48-020",
    "proteinmpnn-v48-020",
    "proteinmpnn",
};

constexpr bool version_is_0_1_0() noexcept {
  return hiko_u::kVersionMajor == 0 && hiko_u::kVersionMinor == 1 &&
         hiko_u::kVersionPatch == 0 && hiko_u::kVersionLabel.empty();
}

void require_0_1_0_registry_lock() {
  if (!version_is_0_1_0()) {
    fail("package registry expectations must be updated for this release");
  }
}

std::string_view detail(const hiko_u::Status status) {
  return status.detail == nullptr ? std::string_view{} : status.detail;
}

bool contains(const std::string_view text, const std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

hiko_w::PackageHandle require_package(
    const hiko_u::Result<hiko_w::PackageHandle>& result,
    const char* message) {
  if (result.status.code != hiko_u::StatusCode::Ok ||
      result.value.descriptor == nullptr || result.value.opaque == nullptr) {
    fail(message);
  }
  return result.value;
}

hiko_u::Span<const hiko_w::PackageRegistryRecord> require_compiled_packages_0_1_0() {
  require_0_1_0_registry_lock();
  const hiko_u::Span<const hiko_w::PackageRegistryRecord> records =
      hiko_w::compiled_packages();
  if (records.data == nullptr || records.size != kExpectedCompiledPackages010) {
    fail(
        "Hikoboshi 0.1.0 must expose MPNN-64, ESM2-8M, and ProteinMPNN "
        "compiled package records");
  }
  return records;
}

void require_default_package_descriptor(
    const hiko_u::PackageDescriptor& descriptor) {
  if (descriptor.identity.package_id != hiko_w::kDefaultMpnnD64ModelName ||
      descriptor.identity.package_family != hiko_w::kDefaultMpnn64ModelFamily ||
      descriptor.identity.package_kind !=
          hiko_u::PackageKind::RegisteredArchitecture ||
      descriptor.identity.aliases.size !=
          (sizeof(kExpectedAliases010) / sizeof(kExpectedAliases010[0]))) {
    fail("registry record must expose the 0.1.0 canonical descriptor shape");
  }
  for (std::size_t index = 0; index < descriptor.identity.aliases.size;
       ++index) {
    if (descriptor.identity.aliases.data[index] != kExpectedAliases010[index]) {
      fail("registry record aliases must match the 0.1.0 package contract");
    }
  }
}

void require_esm2_8m_package_descriptor(
    const hiko_u::PackageDescriptor& descriptor) {
  if (descriptor.identity.package_id != hiko_w::kDefaultEsm2_8mModelName ||
      descriptor.identity.package_family !=
          hiko_w::kDefaultEsm2_8mModelFamily ||
      descriptor.identity.package_kind !=
          hiko_u::PackageKind::RegisteredArchitecture ||
      descriptor.identity.aliases.size !=
          (sizeof(kExpectedEsm2_8mAliases010) /
           sizeof(kExpectedEsm2_8mAliases010[0]))) {
    fail(
        "registry record must expose the ESM2-8M canonical descriptor shape");
  }
  for (std::size_t index = 0; index < descriptor.identity.aliases.size;
       ++index) {
    if (descriptor.identity.aliases.data[index] !=
        kExpectedEsm2_8mAliases010[index]) {
      fail("registry record aliases must match the ESM2-8M package contract");
    }
  }
  if (descriptor.execution.architecture_id !=
      hiko_w::kDefaultEsm2_8mArchitectureId) {
    fail("ESM2-8M descriptor must reference hikoboshi_esm2_v1");
  }
}

void require_proteinmpnn_v48_020_package_descriptor(
    const hiko_u::PackageDescriptor& descriptor) {
  if (descriptor.identity.package_id !=
          hiko_w::kDefaultProteinMpnnV48Eps020ModelName ||
      descriptor.identity.package_family !=
          hiko_w::kDefaultProteinMpnnV48020ModelFamily ||
      descriptor.identity.package_kind != hiko_u::PackageKind::GraphIr ||
      descriptor.identity.aliases.size !=
          (sizeof(kExpectedProteinMpnnV48020Aliases010) /
           sizeof(kExpectedProteinMpnnV48020Aliases010[0]))) {
    fail(
        "registry record must expose the ProteinMPNN v_48_020 descriptor "
        "shape");
  }
  for (std::size_t index = 0; index < descriptor.identity.aliases.size;
       ++index) {
    if (descriptor.identity.aliases.data[index] !=
        kExpectedProteinMpnnV48020Aliases010[index]) {
      fail(
          "registry record aliases must match the ProteinMPNN v_48_020 "
          "package contract");
    }
  }
  if (descriptor.execution.mode != hiko_u::PackageExecutionMode::GraphIr ||
      descriptor.execution.architecture_id !=
          hiko_w::kDefaultProteinMpnnV48Eps020ArchitectureId ||
      descriptor.gaps.family != hiko_w::kInverseFoldingGapFamily ||
      descriptor.gaps.gap_open != 0.0F ||
      descriptor.gaps.gap_extension != 0.0F ||
      descriptor.soft_gaps.family != hiko_w::kInverseFoldingGapFamily ||
      descriptor.soft_gaps.gap_open != 0.0F ||
      descriptor.soft_gaps.gap_extension != 0.0F) {
    fail(
        "ProteinMPNN v_48_020 descriptor must stay on the inverse-folding "
        "reserved path with non-aligner gap sentinels");
  }
}

void require_runtime_available_record(
    const hiko_w::PackageRegistryRecord& record) {
  if (!record.compiled || !record.runtime_available ||
      !record.reason.empty()) {
    fail("compiled hikoboshi-mpnn-d64 must be runtime-available without a reason");
  }
}

void require_esm2_8m_runtime_available_record(
    const hiko_w::PackageRegistryRecord& record) {
  if (!record.compiled || !record.runtime_available ||
      !record.reason.empty()) {
    fail(
        "compiled hikoboshi-esm2-8m must be runtime-available without a "
        "reason once the embedded weights blob ships");
  }
}

void require_proteinmpnn_v48_020_runtime_available_record(
    const hiko_w::PackageRegistryRecord& record) {
  if (!record.compiled || !record.runtime_available ||
      !record.reason.empty()) {
    fail(
        "compiled ProteinMPNN v_48_020 must be runtime-available through the "
        "weights provider without a reason");
  }
}

void test_enumeration_is_static_and_descriptor_backed() {
  const hiko_u::Span<const hiko_w::PackageRegistryRecord> first =
      require_compiled_packages_0_1_0();
  const hiko_u::Span<const hiko_w::PackageRegistryRecord> second =
      hiko_w::compiled_packages();
  if (first.data != second.data || second.size != first.size) {
    fail("compiled package enumeration must return a stable record span");
  }

  const hiko_w::PackageRegistryRecord& mpnn_record = first.data[0];
  if (mpnn_record.package.descriptor == nullptr ||
      mpnn_record.descriptor != mpnn_record.package.descriptor) {
    fail("registry record must expose the canonical descriptor");
  }
  require_default_package_descriptor(*mpnn_record.descriptor);

  const hiko_w::PackageRegistryRecord& esm2_record = first.data[1];
  if (esm2_record.package.descriptor == nullptr ||
      esm2_record.descriptor != esm2_record.package.descriptor) {
    fail("registry record must expose the canonical ESM2-8M descriptor");
  }
  require_esm2_8m_package_descriptor(*esm2_record.descriptor);

  const hiko_w::PackageRegistryRecord& proteinmpnn_record = first.data[2];
  if (proteinmpnn_record.package.descriptor == nullptr ||
      proteinmpnn_record.descriptor !=
          proteinmpnn_record.package.descriptor) {
    fail(
        "registry record must expose the canonical ProteinMPNN v_48_020 "
        "descriptor");
  }
  require_proteinmpnn_v48_020_package_descriptor(
      *proteinmpnn_record.descriptor);
}

void test_availability_fields() {
  const hiko_u::Span<const hiko_w::PackageRegistryRecord> records =
      require_compiled_packages_0_1_0();
  require_runtime_available_record(records.data[0]);
  require_esm2_8m_runtime_available_record(records.data[1]);
  require_proteinmpnn_v48_020_runtime_available_record(records.data[2]);
}

void test_registration_helper_is_plain_record_construction() {
  const int opaque = 0;
  const hiko_u::PackageDescriptor descriptor{};
  const hiko_w::PackageHandle handle{&opaque, &descriptor};

  const hiko_w::PackageRegistryRecord available =
      hiko_w::available_compiled_package_record(handle);
  if (available.package.opaque != &opaque ||
      available.package.descriptor != &descriptor ||
      available.descriptor != &descriptor || !available.compiled ||
      !available.runtime_available || !available.reason.empty()) {
    fail("available compiled package helper must build a plain registry record");
  }

  const hiko_w::PackageRegistryRecord unavailable =
      hiko_w::package_registry_record(handle, true, false, "missing backend");
  if (unavailable.package.opaque != &opaque ||
      unavailable.descriptor != &descriptor || !unavailable.compiled ||
      unavailable.runtime_available ||
      unavailable.reason != std::string_view{"missing backend"}) {
    fail("registry helper must preserve availability fields and reason");
  }
}

void test_canonical_lookup() {
  const hiko_w::PackageHandle package = require_package(
      hiko_w::default_package(hiko_w::kDefaultMpnnD64ModelName),
      "canonical package id must resolve");
  const hiko_u::Span<const hiko_w::PackageRegistryRecord> records =
      require_compiled_packages_0_1_0();
  if (package.descriptor != records.data[0].descriptor ||
      package.opaque != records.data[0].package.opaque) {
    fail("canonical lookup must return the registry package handle");
  }
}

void test_alias_lookup_is_case_insensitive() {
  constexpr std::string_view kAliasInputs[] = {
      "mpnn64",
      "MPNN64",
      "mpnn-64",
      "MpNn-64",
      "Hikoboshi-MPNN-64",
  };

  const hiko_w::PackageHandle canonical = require_package(
      hiko_w::default_package(hiko_w::kDefaultMpnnD64ModelName),
      "canonical package id must resolve before alias parity checks");
  for (const std::string_view alias : kAliasInputs) {
    const hiko_w::PackageHandle package = require_package(
        hiko_w::default_package(alias), "package alias must resolve");
    if (package.descriptor != canonical.descriptor ||
        package.opaque != canonical.opaque ||
        package.descriptor->identity.package_id !=
            hiko_w::kDefaultMpnnD64ModelName) {
      fail("aliases must resolve to the canonical hikoboshi-mpnn-d64 package");
    }
  }
}

void test_unknown_ids_are_rejected_with_available_names() {
  constexpr std::string_view kUnknownIds[] = {
      "",
      "unknown",
      "Hikoboshi-MPNN-65",
      "org/hikoboshi-mpnn-d64",
  };

  for (const std::string_view package_id : kUnknownIds) {
    const hiko_u::Result<hiko_w::PackageHandle> result =
        hiko_w::default_package(package_id);
    const std::string_view message = detail(result.status);
    if (result.status.code != hiko_u::StatusCode::InvalidArgument ||
        result.value.descriptor != nullptr || result.value.opaque != nullptr ||
        !contains(message, "unknown Hikoboshi package id") ||
        !contains(message, "available compiled package IDs/aliases") ||
        !contains(message, hiko_w::kDefaultMpnnD64ModelName) ||
        !contains(message, kExpectedAliases010[0]) ||
        !contains(message, kExpectedAliases010[1]) ||
        !contains(message, kExpectedAliases010[2]) ||
        !contains(message, hiko_w::kDefaultEsm2_8mModelName) ||
        !contains(message, kExpectedEsm2_8mAliases010[0]) ||
        !contains(message, kExpectedEsm2_8mAliases010[1]) ||
        !contains(message, kExpectedEsm2_8mAliases010[2]) ||
        !contains(message, hiko_w::kDefaultProteinMpnnV48Eps020ModelName) ||
        !contains(message, kExpectedProteinMpnnV48020Aliases010[0]) ||
        !contains(message, kExpectedProteinMpnnV48020Aliases010[1]) ||
        !contains(message, kExpectedProteinMpnnV48020Aliases010[2]) ||
        !contains(message, kExpectedProteinMpnnV48020Aliases010[3])) {
      fail(
          "unknown package diagnostics must list available ids and aliases "
          "for every registered package");
    }
  }
}

void test_esm2_8m_default_package_returns_ok() {
  constexpr std::string_view kEsm2_8mIds[] = {
      hiko_w::kDefaultEsm2_8mModelName,
      "esm2-8m",
      "ESM2-8M",
      "esm2_8m",
      "Hikoboshi-ESM2-8M",
  };
  const hiko_u::Span<const hiko_w::PackageRegistryRecord> records =
      require_compiled_packages_0_1_0();
  const hiko_w::PackageRegistryRecord& canonical = records.data[1];
  for (const std::string_view package_id : kEsm2_8mIds) {
    const hiko_w::PackageHandle package = require_package(
        hiko_w::default_package(package_id),
        "ESM2-8M default_package lookup must succeed once the embedded "
        "weights blob ships");
    if (package.descriptor != canonical.descriptor ||
        package.opaque != canonical.package.opaque ||
        package.descriptor->identity.package_id !=
            hiko_w::kDefaultEsm2_8mModelName) {
      fail(
          "ESM2-8M aliases must resolve to the canonical hikoboshi-esm2-8m "
          "registry record");
    }
  }
}

void test_proteinmpnn_v48_020_default_package_returns_ok() {
  constexpr std::string_view kProteinMpnnV48020Ids[] = {
      hiko_w::kDefaultProteinMpnnV48Eps020ModelName,
      "v_48_020",
      "Hikoboshi-ProteinMPNN-v48-020",
      "proteinmpnn-v48-020",
      "PROTEINMPNN-V48-020",
      "proteinmpnn",
  };
  const hiko_u::Span<const hiko_w::PackageRegistryRecord> records =
      require_compiled_packages_0_1_0();
  const hiko_w::PackageRegistryRecord& canonical = records.data[2];
  for (const std::string_view package_id : kProteinMpnnV48020Ids) {
    const hiko_w::PackageHandle package = require_package(
        hiko_w::default_package(package_id),
        "ProteinMPNN v_48_020 default_package lookup must succeed");
    if (package.descriptor != canonical.descriptor ||
        package.opaque != canonical.package.opaque ||
        package.descriptor->identity.package_id !=
            hiko_w::kDefaultProteinMpnnV48Eps020ModelName) {
      fail(
          "ProteinMPNN v_48_020 aliases must resolve to the canonical "
          "registry record");
    }
  }
}

void test_esm2_8m_wrapper_parity() {
  const hiko_w::PackageHandle canonical = require_package(
      hiko_w::default_package(hiko_w::kDefaultEsm2_8mModelName),
      "canonical ESM2-8M package id must resolve for wrapper parity");
  const hiko_w::PackageHandle wrapper = require_package(
      hiko_w::default_esm2_8m_package(),
      "default_esm2_8m_package must preserve package lookup behavior");
  if (canonical.descriptor != wrapper.descriptor ||
      canonical.opaque != wrapper.opaque) {
    fail("default_esm2_8m_package must wrap default_package");
  }

  const hiko_u::Result<hiko_u::WeightsHandle> weights = hiko_w::default_esm2_8m();
  if (weights.status.code != hiko_u::StatusCode::Ok ||
      weights.value.opaque != canonical.opaque ||
      weights.value.view !=
          canonical.descriptor->compatibility_views.weights.view) {
    fail("default_esm2_8m must preserve the package compatibility view");
  }
}

void test_wrapper_parity() {
  const hiko_w::PackageHandle canonical = require_package(
      hiko_w::default_package(hiko_w::kDefaultMpnnD64ModelName),
      "canonical package id must resolve for wrapper parity");
  const hiko_w::PackageHandle wrapper = require_package(
      hiko_w::default_mpnn64_package(),
      "default_mpnn64_package must preserve package lookup behavior");
  if (canonical.descriptor != wrapper.descriptor ||
      canonical.opaque != wrapper.opaque) {
    fail("default_mpnn64_package must wrap default_package");
  }

  const hiko_u::Result<hiko_u::WeightsHandle> weights = hiko_w::default_mpnn64();
  if (weights.status.code != hiko_u::StatusCode::Ok ||
      weights.value.opaque != canonical.opaque ||
      weights.value.view !=
          canonical.descriptor->compatibility_views.weights.view) {
    fail("default_mpnn64 must preserve the package compatibility view");
  }
}

}  // namespace

int main() {
  test_enumeration_is_static_and_descriptor_backed();
  test_availability_fields();
  test_registration_helper_is_plain_record_construction();
  test_canonical_lookup();
  test_alias_lookup_is_case_insensitive();
  test_unknown_ids_are_rejected_with_available_names();
  test_esm2_8m_default_package_returns_ok();
  test_proteinmpnn_v48_020_default_package_returns_ok();
  test_wrapper_parity();
  test_esm2_8m_wrapper_parity();
  return 0;
}
