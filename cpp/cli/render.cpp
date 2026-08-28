#include <hikoboshi/api/all_vs_all.hpp>
#include <hikoboshi/api/results.hpp>
#include <hikoboshi/api/version.hpp>
#include <hikoboshi/errors/format.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <cstddef>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace hikoboshi::cli {
namespace {

std::string format_double(double value) {
  std::ostringstream out;
  out << std::setprecision(6) << value;
  return out.str();
}

std::string format_metric(hikoboshi::universal::MetricValue metric) {
  return hikoboshi::errors::format_metric(metric, 6);
}

hikoboshi::universal::MetricValue invalid_metric(
    hikoboshi::universal::MetricInvalidReason reason) noexcept {
  return {0.0, false, reason};
}

hikoboshi::universal::MetricValue valid_metric(double value) noexcept {
  return {value, true, hikoboshi::universal::MetricInvalidReason::None};
}

hikoboshi::universal::MetricValue sw_per_aligned(
    const hikoboshi::api::PairwiseResult& result) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return invalid_metric(hikoboshi::universal::MetricInvalidReason::
                              ZeroDenominator);
  }
  return valid_metric(result.metrics.raw_sw_score /
                      static_cast<double>(result.path.aligned_pairs));
}

hikoboshi::universal::MetricValue sw_per_length(
    const hikoboshi::api::PairwiseResult& result,
    hikoboshi::universal::MetricValue coverage) noexcept {
  if (result.path.aligned_pairs == 0U) {
    return invalid_metric(hikoboshi::universal::MetricInvalidReason::
                              ZeroDenominator);
  }
  if (!coverage.valid) {
    return invalid_metric(coverage.reason);
  }
  return valid_metric((result.metrics.raw_sw_score * coverage.value) /
                      static_cast<double>(result.path.aligned_pairs));
}

void write_metric_line(std::ostream& out,
                       std::string_view name,
                       hikoboshi::universal::MetricValue value);

void write_metric_line(std::ostream& out,
                       std::string_view name,
                       std::string_view value) {
  out << name << '\t' << value << '\n';
}

void write_metric_line(std::ostream& out,
                       std::string_view name,
                       const char* value) {
  out << name << '\t' << value << '\n';
}

void write_metric_line(std::ostream& out,
                       std::string_view name,
                       const std::string& value) {
  out << name << '\t' << value << '\n';
}

void write_metric_line(std::ostream& out,
                       std::string_view name,
                       hikoboshi::universal::MetricValue value) {
  write_metric_line(out, name, format_metric(value));
}

void render_warning_line(std::ostream& out,
                         const hikoboshi::universal::PackageWarning& warning) {
  out << "hikoboshi: warning";
  if (!warning.code.empty()) {
    out << ": " << warning.code;
  }
  if (!warning.message.empty()) {
    out << ": " << warning.message;
  }
  out << '\n';
}

const char* yes_no(bool value) noexcept {
  return value ? "yes" : "no";
}

const char* backend_name(hikoboshi::universal::Backend backend) noexcept {
  switch (backend) {
    case hikoboshi::universal::Backend::Auto:
      return "auto";
    case hikoboshi::universal::Backend::Scalar:
      return "scalar";
    case hikoboshi::universal::Backend::Sse4:
      return "sse4";
    case hikoboshi::universal::Backend::Avx2:
      return "avx2";
    case hikoboshi::universal::Backend::Avx512:
      return "avx512";
    case hikoboshi::universal::Backend::Neon:
      return "neon";
    case hikoboshi::universal::Backend::Sve:
      return "sve";
    case hikoboshi::universal::Backend::Cuda:
      return "cuda";
    case hikoboshi::universal::Backend::Hip:
      return "hip";
    case hikoboshi::universal::Backend::Metal:
      return "metal";
    case hikoboshi::universal::Backend::Vulkan:
      return "vulkan";
    case hikoboshi::universal::Backend::OpenCl:
      return "opencl";
  }
  return "unknown";
}

const char* package_kind_name(
    hikoboshi::universal::PackageKind package_kind) noexcept {
  switch (package_kind) {
    case hikoboshi::universal::PackageKind::RegisteredArchitecture:
      return "registered_architecture";
    case hikoboshi::universal::PackageKind::GraphIr:
      return "graph_ir";
    case hikoboshi::universal::PackageKind::SubstitutionMatrix:
      return "substitution_matrix";
    case hikoboshi::universal::PackageKind::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* execution_mode_name(
    hikoboshi::universal::PackageExecutionMode mode) noexcept {
  switch (mode) {
    case hikoboshi::universal::PackageExecutionMode::RegisteredArchitecture:
      return "registered_architecture";
    case hikoboshi::universal::PackageExecutionMode::GraphIr:
      return "graph_ir";
    case hikoboshi::universal::PackageExecutionMode::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* score_method_name(hikoboshi::universal::ScoreMethod method) noexcept {
  switch (method) {
    case hikoboshi::universal::ScoreMethod::RawDotV1:
      return "raw_dot_v1";
    case hikoboshi::universal::ScoreMethod::CosineV1:
      return "cosine_v1";
    case hikoboshi::universal::ScoreMethod::SubstitutionLookupV1:
      return "substitution_lookup_v1";
    case hikoboshi::universal::ScoreMethod::DirectScoreMatrixV1:
      return "direct_score_matrix_v1";
    case hikoboshi::universal::ScoreMethod::LearnedPairScorerV1:
      return "learned_pair_scorer_v1";
  }
  return "unknown";
}

const char* score_normalization_name(
    hikoboshi::universal::ScoreNormalization normalization) noexcept {
  switch (normalization) {
    case hikoboshi::universal::ScoreNormalization::None:
      return "none";
    case hikoboshi::universal::ScoreNormalization::L2:
      return "l2";
    case hikoboshi::universal::ScoreNormalization::CalibratedLogOdds:
      return "calibrated_log_odds";
    case hikoboshi::universal::ScoreNormalization::PackageSpecific:
      return "package_specific";
  }
  return "unknown";
}

const char* score_scale_family_name(
    hikoboshi::universal::ScoreScaleFamily family) noexcept {
  switch (family) {
    case hikoboshi::universal::ScoreScaleFamily::RawDot:
      return "raw_dot";
    case hikoboshi::universal::ScoreScaleFamily::CosineUnitless:
      return "cosine_unitless";
    case hikoboshi::universal::ScoreScaleFamily::LogOdds:
      return "log_odds";
    case hikoboshi::universal::ScoreScaleFamily::LearnedLogit:
      return "learned_logit";
    case hikoboshi::universal::ScoreScaleFamily::Unknown:
      return "unknown";
  }
  return "unknown";
}

const char* gap_model_name(hikoboshi::universal::GapModel model) noexcept {
  switch (model) {
    case hikoboshi::universal::GapModel::Affine:
      return "affine";
  }
  return "unknown";
}

const char* gap_convention_name(
    hikoboshi::universal::GapConvention convention) noexcept {
  switch (convention) {
    case hikoboshi::universal::GapConvention::
        GapOpenPlusKMinusOneGapExtension:
      return "gap_open_plus_k_minus_1_gap_extension";
  }
  return "unknown";
}

const char* alignment_algorithm_name(
    hikoboshi::universal::AlignmentAlgorithmId algorithm) noexcept {
  switch (algorithm) {
    case hikoboshi::universal::AlignmentAlgorithmId::HardLocalAffineSwV1:
      return "hard_local_affine_sw_v1";
    case hikoboshi::universal::AlignmentAlgorithmId::GlobalAffineSwV1:
      return "global_affine_sw_v1";
    case hikoboshi::universal::AlignmentAlgorithmId::SemiglobalAffineSwV1:
      return "semiglobal_affine_sw_v1";
    case hikoboshi::universal::AlignmentAlgorithmId::SoftSwV1:
      return "soft_sw_v1";
  }
  return "unknown";
}

bool visible_model_record(
    const hikoboshi::weights::PackageRegistryRecord& record) noexcept {
  return record.descriptor != nullptr &&
         record.descriptor->identity.package_kind ==
             hikoboshi::universal::PackageKind::RegisteredArchitecture;
}

void render_flat_backend_availability(
    std::ostream& out,
    std::string_view key,
    const hikoboshi::api::BackendAvailability& availability) {
  out << key << "_compiled\t" << yes_no(availability.compiled) << '\n';
  out << key << "_runtime_available\t"
      << yes_no(availability.runtime_available) << '\n';
  out << key << "_reason\t" << availability.reason << '\n';
}

void render_flat_gpu_backend_availability(
    std::ostream& out,
    std::string_view key,
    const hikoboshi::api::GpuBackendAvailability& capability) {
  render_flat_backend_availability(out, key, capability.availability);
  out << key << "_device_count\t" << capability.devices.size << '\n';
  for (std::size_t index = 0; index < capability.devices.size; ++index) {
    out << key << "_device_" << index << '\t' << capability.devices.data[index]
        << '\n';
  }
}

void render_path_backend_availability(
    std::ostream& out,
    std::string_view path,
    const hikoboshi::api::BackendAvailability& availability) {
  out << path << ".compiled\t" << yes_no(availability.compiled) << '\n';
  out << path << ".runtime_available\t"
      << yes_no(availability.runtime_available) << '\n';
  out << path << ".reason\t" << availability.reason << '\n';
}

void render_path_gpu_backend_availability(
    std::ostream& out,
    std::string_view path,
    const hikoboshi::api::GpuBackendAvailability& capability) {
  render_path_backend_availability(out, path, capability.availability);
  out << path << ".devices.count\t" << capability.devices.size << '\n';
  for (std::size_t index = 0; index < capability.devices.size; ++index) {
    out << path << ".devices." << index << '\t' << capability.devices.data[index]
        << '\n';
  }
}

}  // namespace

void render_info_backend_summary(
    std::ostream& out,
    const hikoboshi::api::BackendCapabilities& capabilities) {
  out << "backend_default\t" << backend_name(capabilities.default_backend)
      << '\n';
  render_flat_backend_availability(out, "backend_cpu_scalar",
                                   capabilities.cpu.scalar);
  render_flat_backend_availability(out, "backend_cpu_sse4",
                                   capabilities.cpu.sse4);
  render_flat_backend_availability(out, "backend_cpu_avx2",
                                   capabilities.cpu.avx2);
  render_flat_backend_availability(out, "backend_cpu_avx512",
                                   capabilities.cpu.avx512);
  render_flat_backend_availability(out, "backend_cpu_neon",
                                   capabilities.cpu.neon);
  render_flat_backend_availability(out, "backend_cpu_sve", capabilities.cpu.sve);
  render_flat_gpu_backend_availability(out, "backend_gpu_cuda",
                                       capabilities.gpu.cuda);
  render_flat_gpu_backend_availability(out, "backend_gpu_hip",
                                       capabilities.gpu.hip);
  render_flat_gpu_backend_availability(out, "backend_gpu_metal",
                                       capabilities.gpu.metal);
  render_flat_gpu_backend_availability(out, "backend_gpu_vulkan",
                                       capabilities.gpu.vulkan);
  render_flat_gpu_backend_availability(out, "backend_gpu_opencl",
                                       capabilities.gpu.opencl);
  out << "pipeline_pairwise_alignment\t"
      << yes_no(capabilities.pipeline.pairwise_alignment) << '\n';
  out << "pipeline_symmetric_all_vs_all\t"
      << yes_no(capabilities.pipeline.symmetric_all_vs_all) << '\n';
  out << "pipeline_structure_inputs\t"
      << yes_no(capabilities.pipeline.structure_inputs) << '\n';
  out << "pipeline_embedding_inputs\t"
      << yes_no(capabilities.pipeline.embedding_inputs) << '\n';
  out << "threading_compiled\tyes\n";
  out << "threading_default_mode\tauto\n";
}

void render_info_backends(
    std::ostream& out,
    const hikoboshi::api::BackendCapabilities& capabilities) {
  out << "default_backend\t" << backend_name(capabilities.default_backend)
      << '\n';
  render_path_backend_availability(out, "cpu.scalar",
                                   capabilities.cpu.scalar);
  render_path_backend_availability(out, "cpu.sse4", capabilities.cpu.sse4);
  render_path_backend_availability(out, "cpu.avx2", capabilities.cpu.avx2);
  render_path_backend_availability(out, "cpu.avx512",
                                   capabilities.cpu.avx512);
  render_path_backend_availability(out, "cpu.neon", capabilities.cpu.neon);
  render_path_backend_availability(out, "cpu.sve", capabilities.cpu.sve);
  render_path_gpu_backend_availability(out, "gpu.cuda",
                                       capabilities.gpu.cuda);
  render_path_gpu_backend_availability(out, "gpu.hip", capabilities.gpu.hip);
  render_path_gpu_backend_availability(out, "gpu.metal",
                                       capabilities.gpu.metal);
  render_path_gpu_backend_availability(out, "gpu.vulkan",
                                       capabilities.gpu.vulkan);
  render_path_gpu_backend_availability(out, "gpu.opencl",
                                       capabilities.gpu.opencl);
  out << "pipeline.pairwise_alignment\t"
      << yes_no(capabilities.pipeline.pairwise_alignment) << '\n';
  out << "pipeline.symmetric_all_vs_all\t"
      << yes_no(capabilities.pipeline.symmetric_all_vs_all) << '\n';
  out << "pipeline.structure_inputs\t"
      << yes_no(capabilities.pipeline.structure_inputs) << '\n';
  out << "pipeline.embedding_inputs\t"
      << yes_no(capabilities.pipeline.embedding_inputs) << '\n';
  out << "threading.compiled\tyes\n";
  out << "threading.default_mode\tauto\n";
}

const hikoboshi::weights::WeightManifestView& manifest_for_package(
    std::string_view package_id) noexcept {
  if (package_id == hikoboshi::weights::kDefaultEsm2_8mModelName) {
    return hikoboshi::weights::default_esm2_8m_manifest();
  }
  return hikoboshi::weights::default_mpnn_d64_manifest();
}

void render_info_models(std::ostream& out) {
  const hikoboshi::universal::Span<
      const hikoboshi::weights::PackageRegistryRecord>
      records = hikoboshi::weights::compiled_packages();

  std::size_t visible_count = 0;
  for (std::size_t index = 0; index < records.size; ++index) {
    if (visible_model_record(records.data[index])) {
      ++visible_count;
    }
  }

  out << "models.count\t" << visible_count << '\n';
  std::size_t model_index = 0;
  for (std::size_t index = 0; index < records.size; ++index) {
    const hikoboshi::weights::PackageRegistryRecord& record = records.data[index];
    if (!visible_model_record(record)) {
      continue;
    }

    const hikoboshi::universal::PackageDescriptor& descriptor =
        *record.descriptor;
    const hikoboshi::weights::WeightManifestView& manifest =
        manifest_for_package(descriptor.identity.package_id);
    const std::string prefix = "models." + std::to_string(model_index);
    const std::size_t hidden_dimension =
        descriptor.compatibility_views.weights.view == nullptr
            ? manifest.hidden_dimension
            : descriptor.compatibility_views.weights.view->metadata
                  .hidden_dimension;

    out << prefix << ".package_id\t" << descriptor.identity.package_id << '\n';
    out << prefix << ".aliases.count\t" << descriptor.identity.aliases.size
        << '\n';
    for (std::size_t alias_index = 0;
         alias_index < descriptor.identity.aliases.size; ++alias_index) {
      out << prefix << ".aliases." << alias_index << '\t'
          << descriptor.identity.aliases.data[alias_index] << '\n';
    }
    out << prefix << ".family\t" << descriptor.identity.package_family << '\n';
    out << prefix << ".version\t" << descriptor.identity.package_version
        << '\n';
    out << prefix << ".package_kind\t"
        << package_kind_name(descriptor.identity.package_kind) << '\n';
    out << prefix << ".execution_mode\t"
        << execution_mode_name(descriptor.execution.mode) << '\n';
    out << prefix << ".architecture_id\t"
        << descriptor.execution.architecture_id << '\n';
    out << prefix << ".hidden_dimension\t" << hidden_dimension << '\n';

    out << prefix << ".scoring.method\t"
        << score_method_name(descriptor.scoring.method) << '\n';
    out << prefix << ".scoring.similarity\t" << manifest.similarity << '\n';
    out << prefix << ".scoring.normalization\t"
        << score_normalization_name(
               descriptor.scoring.semantics.normalization)
        << '\n';
    out << prefix << ".scoring.scale_family\t"
        << score_scale_family_name(descriptor.scoring.semantics.scale_family)
        << '\n';

    out << prefix << ".gaps.family\t" << descriptor.gaps.family << '\n';
    out << prefix << ".gaps.model\t"
        << gap_model_name(descriptor.gaps.model) << '\n';
    out << prefix << ".gaps.gap_open\t" << descriptor.gaps.gap_open << '\n';
    out << prefix << ".gaps.gap_extension\t"
        << descriptor.gaps.gap_extension << '\n';
    out << prefix << ".gaps.convention\t"
        << gap_convention_name(descriptor.gaps.convention) << '\n';
    out << prefix << ".gaps.calibrated_for_score_method\t"
        << score_method_name(descriptor.gaps.calibrated_for_score_method)
        << '\n';
    out << prefix << ".soft_gaps.family\t" << descriptor.soft_gaps.family
        << '\n';
    out << prefix << ".soft_gaps.model\t"
        << gap_model_name(descriptor.soft_gaps.model) << '\n';
    out << prefix << ".soft_gaps.gap_open\t"
        << descriptor.soft_gaps.gap_open << '\n';
    out << prefix << ".soft_gaps.gap_extension\t"
        << descriptor.soft_gaps.gap_extension << '\n';
    out << prefix << ".soft_gaps.convention\t"
        << gap_convention_name(descriptor.soft_gaps.convention) << '\n';
    out << prefix << ".soft_gaps.calibrated_for_score_method\t"
        << score_method_name(
               descriptor.soft_gaps.calibrated_for_score_method)
        << '\n';
    out << prefix << ".alignment.algorithm\t"
        << alignment_algorithm_name(descriptor.alignment.algorithm) << '\n';

    out << prefix << ".checksum.algorithm\t" << manifest.checksum_algorithm
        << '\n';
    out << prefix << ".checksum.package\t" << manifest.checksum << '\n';
    out << prefix << ".checksum.source_checkpoint\t"
        << manifest.source_checkpoint_checksum << '\n';
    out << prefix << ".provenance.source_checkpoint\t"
        << manifest.source_checkpoint << '\n';
    out << prefix << ".provenance.generation_tool\t"
        << manifest.generation_tool << '\n';
    out << prefix << ".provenance.generation_tool_version\t"
        << manifest.generation_tool_version << '\n';
    out << prefix << ".provenance.generation_date\t"
        << manifest.generation_date << '\n';
    out << prefix << ".provenance.validation_status\t"
        << manifest.validation_status << '\n';
    out << prefix << ".provenance.status\t" << manifest.provenance_status
        << '\n';

    out << prefix << ".availability.compiled\t" << yes_no(record.compiled)
        << '\n';
    out << prefix << ".availability.runtime_available\t"
        << yes_no(record.runtime_available) << '\n';
    out << prefix << ".availability.reason\t" << record.reason << '\n';
    ++model_index;
  }
}

void render_encode_summary(std::ostream& out,
                           std::string_view input_mode,
                           std::string_view input,
                           const hikoboshi::api::EncodeResult& result) {
  write_metric_line(out, "command", "encode");
  write_metric_line(out, "input_mode", input_mode);
  write_metric_line(out, "input", input);
  write_metric_line(out, "residue_count",
                    std::to_string(result.embedding.residue_count));
  write_metric_line(out, "embedding_dimension",
                    std::to_string(result.embedding.dimension));
}

void render_pairwise_summary(std::ostream& out,
                             std::string_view input_mode,
                             std::string_view query,
                             std::string_view target,
                             const hikoboshi::api::PairwiseResult& result,
                             bool include_dual_score_schema) {
  write_metric_line(out, "command", "pairwise");
  write_metric_line(out, "input_mode", input_mode);
  write_metric_line(out, "query", query);
  write_metric_line(out, "target", target);
  write_metric_line(out, "raw_sw_score",
                    format_double(result.metrics.raw_sw_score));
  if (include_dual_score_schema) {
    write_metric_line(out, "soft_sw_score", result.metrics.soft_sw_score);
    write_metric_line(out, "sw_per_query_len",
                      sw_per_length(result, result.metrics.coverage_query));
    write_metric_line(out, "sw_per_target_len",
                      sw_per_length(result, result.metrics.coverage_target));
    write_metric_line(out, "sw_per_aligned", sw_per_aligned(result));
  }
  write_metric_line(out, "aligned_pairs",
                    std::to_string(result.path.aligned_pairs));
  write_metric_line(out, "coverage_query", result.metrics.coverage_query);
  write_metric_line(out, "coverage_target", result.metrics.coverage_target);
  write_metric_line(out, "coverage_mean", result.metrics.coverage_mean);
  write_metric_line(out, "identity", result.metrics.identity);
  write_metric_line(out, "rmsd", result.metrics.rmsd);
  write_metric_line(out, "tm_score_query", result.metrics.tm_score_query);
  write_metric_line(out, "tm_score_target", result.metrics.tm_score_target);
  write_metric_line(out, "lddt", result.metrics.lddt);
  write_metric_line(out, "lddt_byA", result.metrics.lddt_byA);
  write_metric_line(out, "lddt_byB", result.metrics.lddt_byB);
  write_metric_line(out, "lddt_aln", result.metrics.lddt_aln);
  write_metric_line(out, "coverage_byA", result.metrics.coverage_byA);
  write_metric_line(out, "coverage_byB", result.metrics.coverage_byB);
  write_metric_line(out, "ecs", result.metrics.ecs);
}

void render_pairwise_warnings(std::ostream& out,
                              const hikoboshi::api::PairwiseResult& result) {
  for (const hikoboshi::universal::PackageWarning& warning : result.warnings) {
    render_warning_line(out, warning);
  }
}

void render_gap_override_warning(std::ostream& out) {
  render_warning_line(
      out,
      {hikoboshi::universal::PackageWarningKind::GapDefaultsOverridden,
       hikoboshi::universal::PackageValidationStage::GapModelDefaults,
       "gap_defaults_overridden",
       "user-provided affine gap values override hikoboshi-mpnn-d64 hard-SW calibrated defaults gap_open=-1.40000 and gap_extension=-0.150000"});
}

void render_gap_override_warning(std::ostream& out, std::string_view message) {
  render_warning_line(
      out,
      {hikoboshi::universal::PackageWarningKind::GapDefaultsOverridden,
       hikoboshi::universal::PackageValidationStage::GapModelDefaults,
       "gap_defaults_overridden",
       message});
}

void render_all_vs_all_summary(
    std::ostream& out,
    const hikoboshi::api::AllVsAllResult& result,
    const std::vector<std::string>& pair_ids,
    const std::vector<std::string>& fasta_paths,
    const std::vector<std::string>& pdb_paths,
    bool include_dual_score_schema) {
  out << "query_index\ttarget_index\tpair_id\traw_sw_score";
  if (include_dual_score_schema) {
    out << "\tsoft_sw_score\tsw_per_query_len\tsw_per_target_len"
        << "\tsw_per_aligned";
  }
  out << "\taligned_pairs"
      << "\tcoverage_query\tcoverage_target\tcoverage_mean\tidentity\trmsd"
      << "\ttm_score_query\ttm_score_target"
      << "\tlddt\tlddt_byA\tlddt_byB\tlddt_aln\tcoverage_byA\tcoverage_byB"
      << "\tecs\tfasta_path\tpdb_path\n";
  for (std::size_t index = 0; index < result.records.size(); ++index) {
    const hikoboshi::api::PairwiseResultRecord& record = result.records[index];
    const hikoboshi::api::PairwiseMetrics& metrics = record.result.metrics;
    const std::string empty;
    const std::string& pair_id =
        index < pair_ids.size() ? pair_ids[index] : empty;
    const std::string& fasta_path =
        index < fasta_paths.size() ? fasta_paths[index] : empty;
    const std::string& pdb_path =
        index < pdb_paths.size() ? pdb_paths[index] : empty;
    out << record.query_index << '\t' << record.target_index << '\t'
        << pair_id << '\t' << format_double(metrics.raw_sw_score);
    if (include_dual_score_schema) {
      out << '\t' << format_metric(metrics.soft_sw_score) << '\t'
          << format_metric(sw_per_length(record.result,
                                         metrics.coverage_query))
          << '\t'
          << format_metric(sw_per_length(record.result,
                                         metrics.coverage_target))
          << '\t' << format_metric(sw_per_aligned(record.result));
    }
    out << '\t'
        << record.result.path.aligned_pairs << '\t'
        << format_metric(metrics.coverage_query) << '\t'
        << format_metric(metrics.coverage_target) << '\t'
        << format_metric(metrics.coverage_mean) << '\t'
        << format_metric(metrics.identity) << '\t'
        << format_metric(metrics.rmsd) << '\t'
        << format_metric(metrics.tm_score_query) << '\t'
        << format_metric(metrics.tm_score_target) << '\t'
        << format_metric(metrics.lddt) << '\t'
        << format_metric(metrics.lddt_byA) << '\t'
        << format_metric(metrics.lddt_byB) << '\t'
        << format_metric(metrics.lddt_aln) << '\t'
        << format_metric(metrics.coverage_byA) << '\t'
        << format_metric(metrics.coverage_byB) << '\t'
        << format_metric(metrics.ecs) << '\t' << fasta_path << '\t'
        << pdb_path << '\n';
  }
}

}  // namespace hikoboshi::cli
