#include <hikoboshi/modules/mpnn.hpp>
#include <hikoboshi/weights/manifest.hpp>
#include <hikoboshi/weights/provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace hiko_m = hikoboshi::modules;
namespace hiko_d = hikoboshi::modules::detail;
namespace hiko_u = hikoboshi::universal;
namespace hiko_w = hikoboshi::weights;

namespace {

struct Args {
  std::string pid;
  std::filesystem::path coords_bin;
  std::filesystem::path atom_sources_bin;
  std::filesystem::path dump_dir;
  std::size_t length = 0;
  std::size_t coords_atom_count = 4;
};

struct OwnedWorkspace {
  std::vector<float> ca_coordinates;
  std::vector<float> residue_features;
  std::vector<std::int32_t> neighbor_indices;
  std::vector<float> neighbor_squared_distances;
  std::vector<float> rbf_features;
  std::vector<float> residue_state;
  std::vector<float> gathered_state;
  std::vector<float> edge_state;
  std::vector<float> message_state;
  std::vector<float> projected_message_state;
  std::vector<float> residue_scratch;
  std::vector<float> ffn_hidden;
  hiko_d::Mpnn64Workspace view{};
};

[[noreturn]] void fail(const std::string& detail) {
  std::cerr << "mpnn64_real_parity: " << detail << "\n";
  std::exit(1);
}

std::size_t checked_product(const hiko_m::Mpnn64IntermediateTensor& tensor) {
  std::size_t count = 1;
  for (std::size_t index = 0; index < tensor.rank; ++index) {
    count *= tensor.shape[index];
  }
  return count;
}

std::string sanitized(std::string name) {
  for (char& c : name) {
    if (!(c == '.' || c == '_' || c == '-' ||
          (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
          (c >= 'a' && c <= 'z'))) {
      c = '_';
    }
  }
  return name;
}

void write_tensor(const hiko_m::Mpnn64IntermediateTensor& tensor,
                  void* user_data) noexcept {
  try {
    const auto* dump_dir =
        static_cast<const std::filesystem::path*>(user_data);
    const std::string stem = sanitized(tensor.name);
    const bool is_float =
        tensor.dtype == hiko_m::Mpnn64IntermediateDtype::Float32;
    const std::filesystem::path data_path =
        *dump_dir / (stem + (is_float ? ".f32" : ".i32"));
    const std::filesystem::path meta_path = *dump_dir / (stem + ".json");

    std::ofstream data(data_path, std::ios::binary);
    const std::size_t count = checked_product(tensor);
    const std::size_t bytes =
        count * (is_float ? sizeof(float) : sizeof(std::int32_t));
    data.write(static_cast<const char*>(tensor.data),
               static_cast<std::streamsize>(bytes));
    data.close();

    std::ofstream meta(meta_path);
    meta << "{\n";
    meta << "  \"name\": \"" << tensor.name << "\",\n";
    meta << "  \"dtype\": \"" << (is_float ? "float32" : "int32") << "\",\n";
    meta << "  \"data\": \"" << data_path.filename().string() << "\",\n";
    meta << "  \"shape\": [";
    for (std::size_t index = 0; index < tensor.rank; ++index) {
      if (index != 0) {
        meta << ", ";
      }
      meta << tensor.shape[index];
    }
    meta << "]\n";
    meta << "}\n";
  } catch (...) {
  }
}

Args parse_args(int argc, char** argv) {
  Args args{};
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--pid" && index + 1 < argc) {
      args.pid = argv[++index];
    } else if (flag == "--coords-bin" && index + 1 < argc) {
      args.coords_bin = argv[++index];
    } else if (flag == "--atom-sources-bin" && index + 1 < argc) {
      args.atom_sources_bin = argv[++index];
    } else if (flag == "--coords-atom-count" && index + 1 < argc) {
      args.coords_atom_count =
          static_cast<std::size_t>(std::stoul(argv[++index]));
    } else if (flag == "--dump-dir" && index + 1 < argc) {
      args.dump_dir = argv[++index];
    } else if (flag == "--length" && index + 1 < argc) {
      args.length = static_cast<std::size_t>(std::stoul(argv[++index]));
    } else {
      fail("unknown or incomplete argument: " + flag);
    }
  }
  if (args.pid.empty() || args.coords_bin.empty() || args.dump_dir.empty() ||
      args.length == 0) {
    fail("usage: --pid ID --length L --coords-bin PATH --dump-dir PATH");
  }
  if (args.coords_atom_count != 4 && args.coords_atom_count != 5) {
    fail("--coords-atom-count must be 4 or 5");
  }
  if (args.coords_atom_count == 5 && args.atom_sources_bin.empty()) {
    fail("--atom-sources-bin is required when --coords-atom-count is 5");
  }
  return args;
}

std::vector<float> read_coords(const std::filesystem::path& path,
                               std::size_t length,
                               std::size_t atom_count) {
  std::vector<float> coords(length * atom_count * 3, 0.0F);
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("could not open coords input: " + path.string());
  }
  input.read(reinterpret_cast<char*>(coords.data()),
             static_cast<std::streamsize>(coords.size() * sizeof(float)));
  if (input.gcount() !=
      static_cast<std::streamsize>(coords.size() * sizeof(float))) {
    fail("coords input has unexpected size: " + path.string());
  }
  return coords;
}

std::vector<hiko_u::AtomSource> read_atom_sources(
    const std::filesystem::path& path,
    std::size_t length) {
  std::vector<unsigned char> bytes(length * hiko_u::kCanonicalAtomCount, 0);
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("could not open atom_sources input: " + path.string());
  }
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    fail("atom_sources input has unexpected size: " + path.string());
  }
  std::vector<hiko_u::AtomSource> sources(bytes.size(), hiko_u::AtomSource::Missing);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    if (bytes[index] > static_cast<unsigned char>(hiko_u::AtomSource::Virtual)) {
      fail("atom_sources input contains invalid enum value");
    }
    sources[index] = static_cast<hiko_u::AtomSource>(bytes[index]);
  }
  return sources;
}

std::size_t coord4_offset(std::size_t residue,
                          std::size_t atom,
                          std::size_t axis) noexcept {
  return (residue * 4 + atom) * 3 + axis;
}

std::size_t coord5_offset(std::size_t residue,
                          std::size_t atom,
                          std::size_t axis) noexcept {
  return (residue * hiko_u::kCanonicalAtomCount + atom) *
             hiko_u::kCoordinateAxisCount +
         axis;
}

bool present_atom(const std::vector<float>& coords4,
                  std::size_t residue,
                  std::size_t atom) noexcept {
  float total = 0.0F;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    total += std::fabs(coords4[coord4_offset(residue, atom, axis)]);
  }
  return total > 0.1F;
}

std::size_t coord_input_offset(std::size_t residue,
                               std::size_t atom_count,
                               std::size_t atom,
                               std::size_t axis) noexcept {
  return (residue * atom_count + atom) * 3 + axis;
}

std::array<float, 3> virtual_cb(const std::vector<float>& coords4,
                                std::size_t residue) noexcept {
  const float* n = coords4.data() + coord4_offset(residue, 0, 0);
  const float* ca = coords4.data() + coord4_offset(residue, 1, 0);
  const float* c = coords4.data() + coord4_offset(residue, 2, 0);
  const float b[3] = {ca[0] - n[0], ca[1] - n[1], ca[2] - n[2]};
  const float cc[3] = {c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]};
  const float a[3] = {
      b[1] * cc[2] - b[2] * cc[1],
      b[2] * cc[0] - b[0] * cc[2],
      b[0] * cc[1] - b[1] * cc[0],
  };
  return {
      ca[0] - 0.58273431F * a[0] + 0.56802827F * b[0] -
          0.54067466F * cc[0],
      ca[1] - 0.58273431F * a[1] + 0.56802827F * b[1] -
          0.54067466F * cc[1],
      ca[2] - 0.58273431F * a[2] + 0.56802827F * b[2] -
          0.54067466F * cc[2],
  };
}

void prepare_coords5(const std::vector<float>& coords4,
                     std::size_t length,
                     std::vector<float>& coords5,
                     std::vector<hiko_u::AtomSource>& atom_sources,
                     std::vector<float>& cb_dump,
                     std::vector<float>& atom_mask) {
  coords5.assign(length * hiko_u::kCanonicalAtomCount *
                     hiko_u::kCoordinateAxisCount,
                 0.0F);
  atom_sources.assign(length * hiko_u::kCanonicalAtomCount,
                      hiko_u::AtomSource::Missing);
  cb_dump.assign(length * 3, 0.0F);
  atom_mask.assign(length * hiko_u::kCanonicalAtomCount, 0.0F);

  for (std::size_t residue = 0; residue < length; ++residue) {
    for (std::size_t atom = 0; atom < 4; ++atom) {
      const bool present = present_atom(coords4, residue, atom);
      atom_sources[residue * hiko_u::kCanonicalAtomCount + atom] =
          present ? hiko_u::AtomSource::Observed : hiko_u::AtomSource::Missing;
      atom_mask[residue * hiko_u::kCanonicalAtomCount + atom] =
          present ? 1.0F : 0.0F;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        coords5[coord5_offset(residue, atom, axis)] =
            coords4[coord4_offset(residue, atom, axis)];
      }
    }
    const bool cb_present = present_atom(coords4, residue, 0) &&
                            present_atom(coords4, residue, 1) &&
                            present_atom(coords4, residue, 2);
    const std::array<float, 3> cb = virtual_cb(coords4, residue);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      cb_dump[residue * 3 + axis] = cb_present ? cb[axis] : 0.0F;
      coords5[coord5_offset(residue, 4, axis)] = cb_dump[residue * 3 + axis];
    }
    atom_sources[residue * hiko_u::kCanonicalAtomCount + 4] =
        cb_present ? hiko_u::AtomSource::Virtual : hiko_u::AtomSource::Missing;
    atom_mask[residue * hiko_u::kCanonicalAtomCount + 4] =
        cb_present ? 1.0F : 0.0F;
  }
}

void prepare_coords5_from_fixture(
    const std::vector<float>& coords_input,
    const std::vector<hiko_u::AtomSource>& sources_input,
    std::size_t length,
    std::vector<float>& coords5,
    std::vector<hiko_u::AtomSource>& atom_sources,
    std::vector<float>& cb_dump,
    std::vector<float>& atom_mask) {
  coords5.assign(length * hiko_u::kCanonicalAtomCount *
                     hiko_u::kCoordinateAxisCount,
                 0.0F);
  atom_sources = sources_input;
  cb_dump.assign(length * 3, 0.0F);
  atom_mask.assign(length * hiko_u::kCanonicalAtomCount, 0.0F);

  if (atom_sources.size() != length * hiko_u::kCanonicalAtomCount) {
    fail("atom_sources length does not match residue count");
  }

  for (std::size_t residue = 0; residue < length; ++residue) {
    for (std::size_t atom = 0; atom < hiko_u::kCanonicalAtomCount; ++atom) {
      const auto source =
          atom_sources[residue * hiko_u::kCanonicalAtomCount + atom];
      atom_mask[residue * hiko_u::kCanonicalAtomCount + atom] =
          source == hiko_u::AtomSource::Missing ? 0.0F : 1.0F;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        coords5[coord5_offset(residue, atom, axis)] =
            coords_input[coord_input_offset(
                residue, hiko_u::kCanonicalAtomCount, atom, axis)];
      }
    }
    if (atom_sources[residue * hiko_u::kCanonicalAtomCount + 4] !=
        hiko_u::AtomSource::Missing) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        cb_dump[residue * 3 + axis] = coords5[coord5_offset(residue, 4, axis)];
      }
    }
  }
}

OwnedWorkspace make_workspace(const hiko_d::Mpnn64MemoryPlan& plan) {
  OwnedWorkspace owned{};
  owned.ca_coordinates.resize(hiko_d::mpnn64_ca_coordinate_count(plan));
  owned.residue_features.resize(hiko_d::mpnn64_residue_feature_count(plan));
  owned.neighbor_indices.resize(hiko_d::mpnn64_neighbor_slot_count(plan));
  owned.neighbor_squared_distances.resize(hiko_d::mpnn64_neighbor_slot_count(plan));
  owned.rbf_features.resize(hiko_d::mpnn64_neighbor_rbf_count(plan));
  owned.residue_state.resize(hiko_d::mpnn64_residue_hidden_count(plan));
  owned.gathered_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.edge_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.message_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.projected_message_state.resize(hiko_d::mpnn64_neighbor_hidden_count(plan));
  owned.residue_scratch.resize(hiko_d::mpnn64_residue_hidden_count(plan));
  owned.ffn_hidden.resize(hiko_d::mpnn64_ffn_hidden_count(plan));
  owned.view = {
      plan,
      {owned.ca_coordinates.data(), owned.ca_coordinates.size()},
      {owned.residue_features.data(), owned.residue_features.size()},
      {owned.neighbor_indices.data(), owned.neighbor_indices.size()},
      {owned.neighbor_squared_distances.data(),
       owned.neighbor_squared_distances.size()},
      {owned.rbf_features.data(), owned.rbf_features.size()},
      {owned.residue_state.data(), owned.residue_state.size()},
      {owned.gathered_state.data(), owned.gathered_state.size()},
      {owned.edge_state.data(), owned.edge_state.size()},
      {owned.message_state.data(), owned.message_state.size()},
      {owned.projected_message_state.data(),
       owned.projected_message_state.size()},
      {owned.residue_scratch.data(), owned.residue_scratch.size()},
      {owned.ffn_hidden.data(), owned.ffn_hidden.size()},
  };
  return owned;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef HIKOBOSHI_BENCH_MPNN_DUMP
  (void)argc;
  (void)argv;
  fail("this executable must be built with HIKOBOSHI_BENCH_MPNN_DUMP");
#else
  // Applicable parity mode: strict.
  //
  // This harness dumps intermediate Hikoboshi-MPNN-64 tensors and compares
  // them to PyTorch CUDA-cuBLAS reference tensors at tighter than 1e-4
  // tolerance. Fast-mode GEMM stays within the public 1e-4 contract but
  // produces a different reduction tree, so any drift introduced here
  // belongs to MPNN encoder parity rather than GEMM kernel choice. Force
  // strict regardless of the build-time `hikoboshi_gemm_parity_mode`.
  setenv("HIKOBOSHI_GEMM_PARITY_MODE", "strict", 1);

  Args args = parse_args(argc, argv);
  std::filesystem::create_directories(args.dump_dir);

  const std::vector<float> coords_input =
      read_coords(args.coords_bin, args.length, args.coords_atom_count);
  std::vector<float> coords5;
  std::vector<hiko_u::AtomSource> atom_sources;
  std::vector<float> cb_dump;
  std::vector<float> atom_mask;
  if (args.coords_atom_count == 5) {
    const std::vector<hiko_u::AtomSource> sources_input =
        read_atom_sources(args.atom_sources_bin, args.length);
    prepare_coords5_from_fixture(coords_input, sources_input, args.length,
                                 coords5, atom_sources, cb_dump, atom_mask);
  } else {
    prepare_coords5(coords_input, args.length, coords5, atom_sources, cb_dump,
                    atom_mask);
  }

  const hiko_w::WeightManifestView& manifest = hiko_w::default_mpnn_d64_manifest();
  const hiko_m::Mpnn64Descriptor descriptor{manifest.hidden_dimension,
                                         manifest.neighbor_count,
                                         manifest.rbf_count,
                                         manifest.layer_count,
                                         manifest.message_scale};
  const auto weights = hiko_w::default_mpnn_d64();
  if (weights.status.code != hiko_u::StatusCode::Ok) {
    fail("default Hikoboshi-MPNN-64 weights failed to prepare");
  }
  const auto* prepared =
      static_cast<const hiko_d::Mpnn64Weights*>(weights.value.opaque);
  if (prepared == nullptr) {
    fail("default Hikoboshi-MPNN-64 weights opaque pointer is null");
  }

  const hiko_d::Mpnn64MemoryPlan plan{args.length,
                                   descriptor.hidden_dimension,
                                   descriptor.neighbor_count,
                                   descriptor.rbf_count,
                                   descriptor.layer_count};
  OwnedWorkspace workspace = make_workspace(plan);
  std::vector<float> embeddings(args.length * descriptor.hidden_dimension,
                                0.0F);
  std::vector<std::int32_t> residue_indices(args.length, 0);
  std::vector<std::int32_t> chain_labels(args.length, 0);
  for (std::size_t index = 0; index < args.length; ++index) {
    residue_indices[index] = static_cast<std::int32_t>(index);
  }

  const hiko_m::Mpnn64IntermediateTensor cb_tensor{
      "input.virtual_cb",
      cb_dump.data(),
      hiko_m::Mpnn64IntermediateDtype::Float32,
      2,
      {args.length, 3, 0, 0},
  };
  write_tensor(cb_tensor, &args.dump_dir);
  const hiko_m::Mpnn64IntermediateTensor mask_tensor{
      "input.atom_mask",
      atom_mask.data(),
      hiko_m::Mpnn64IntermediateDtype::Float32,
      2,
      {args.length, hiko_u::kCanonicalAtomCount, 0, 0},
  };
  write_tensor(mask_tensor, &args.dump_dir);

  hiko_m::Mpnn64ForwardRequest request{};
  request.coordinates = coords5.data();
  request.atom_sources = atom_sources.data();
  request.residue_count = args.length;
  request.descriptor = descriptor;
  request.weights = prepared;
  request.workspace = &workspace.view;
  request.residue_indices = residue_indices.data();
  request.chain_labels = chain_labels.data();
  request.intermediate_dumper.callback = write_tensor;
  request.intermediate_dumper.user_data = &args.dump_dir;

  const hiko_u::Status status = hiko_m::mpnn64_forward_scalar(
      request, {embeddings.data(), args.length, descriptor.hidden_dimension});
  if (status.code != hiko_u::StatusCode::Ok) {
    fail(std::string("MPNN forward failed: ") + status.detail);
  }
  return 0;
#endif
}
