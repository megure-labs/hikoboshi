#ifndef HIKOBOSHI_UNIVERSAL_PACKAGE_VALIDATION_CODES_HPP
#define HIKOBOSHI_UNIVERSAL_PACKAGE_VALIDATION_CODES_HPP

/// @file
/// Stable rejection-code strings for the staged package validation pipeline.
///
/// The codes here are the canonical vocabulary from the model/scoring
/// package decision report
/// (`docs/architecture/hikoboshi_model_package_execution_schema_claude.json`
/// at `capability_validation_recommendation.ten_stage_pipeline`). They are
/// emitted by the staged validator implemented in
/// `cpp/dispatch/registry/validation_core.cpp` and consumed by package
/// validation reports through `PackageValidationDiagnostic::code`.
///
/// Each code is a `constexpr std::string_view` with static storage so report
/// diagnostics remain allocation-free; the strings outlive any caller-owned
/// validation buffer.
///
/// Additivity rule: new codes may be appended for new failure modes, but
/// existing strings must remain stable. Consumers that match on a code (tests,
/// CLI/Python diagnostics, downstream tooling) can rely on the spelling here.

#include <string_view>

namespace hikoboshi::universal {

/// Canonical rejection-code strings for the ten-stage package validator.
namespace package_validation_codes {

// Stage 1 — schema_validation.
inline constexpr std::string_view kPackageSchemaVersionUnsupported{
    "package_schema_version_unsupported"};
inline constexpr std::string_view kArchitectureKindUnknown{
    "architecture_kind_unknown"};
inline constexpr std::string_view kArchitectureParamsInvalid{
    "architecture_params_invalid"};
inline constexpr std::string_view kTensorRoleUnknown{"tensor_role_unknown"};

// Stage 2 — architecture_binding.
inline constexpr std::string_view kArchitectureKindNotRegistered{
    "architecture_kind_not_registered"};

// Stage 3 — capability_handshake.
inline constexpr std::string_view kCapabilityHandshakeFailed{
    "capability_handshake_failed"};
inline constexpr std::string_view kBackendUnavailable{"backend_unavailable"};
inline constexpr std::string_view kDtypeUnsupportedOnBackend{
    "dtype_unsupported_on_backend"};
inline constexpr std::string_view kLayoutUnsupportedOnBackend{
    "layout_unsupported_on_backend"};

// Stage 4 — tensor_table_validation.
inline constexpr std::string_view kTensorSlotUnknown{"tensor_slot_unknown"};
inline constexpr std::string_view kTensorSlotShapeMismatch{
    "tensor_slot_shape_mismatch"};
inline constexpr std::string_view kTensorSlotDtypeMismatch{
    "tensor_slot_dtype_mismatch"};
inline constexpr std::string_view kTensorSlotLayoutMismatch{
    "tensor_slot_layout_mismatch"};
inline constexpr std::string_view kTensorRequiredSlotMissing{
    "tensor_required_slot_missing"};

// Stage 5 — prepared_state_plan_check.
inline constexpr std::string_view kPreparedStatePlanInvalid{
    "prepared_state_plan_invalid"};

// Stage 6 — scoring_binding.
inline constexpr std::string_view kScoringKindNotRegistered{
    "scoring_kind_not_registered"};
inline constexpr std::string_view kScoringInputIncompatibleWithArchitecture{
    "scoring_input_incompatible_with_architecture"};

// Stage 7 — alignment_binding.
inline constexpr std::string_view kAlignmentKindNotRegistered{
    "alignment_kind_not_registered"};
inline constexpr std::string_view kGapFamilyNotSupportedByAlignment{
    "gap_family_not_supported_by_alignment"};

// Stage 8 — io_contract_binding.
inline constexpr std::string_view kIoContractIncompatible{
    "io_contract_incompatible"};

// Stage 9 — provenance_check.
inline constexpr std::string_view kPayloadChecksumMismatch{
    "payload_checksum_mismatch"};
inline constexpr std::string_view kTensorChecksumMismatch{
    "tensor_checksum_mismatch"};

}  // namespace package_validation_codes

}  // namespace hikoboshi::universal

#endif  // HIKOBOSHI_UNIVERSAL_PACKAGE_VALIDATION_CODES_HPP
