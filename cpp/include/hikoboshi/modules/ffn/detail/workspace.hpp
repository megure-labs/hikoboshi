#ifndef HIKOBOSHI_MODULES_FFN_DETAIL_WORKSPACE_HPP
#define HIKOBOSHI_MODULES_FFN_DETAIL_WORKSPACE_HPP

/// @file
/// Caller-owned workspace view for the FFN template family.
///
/// Hikoboshi compound-module workspaces are never allocated by the module: the
/// caller supplies a single intermediate buffer the inline body writes into
/// for the post-`w_in` activation output. Architectures slice this buffer
/// from their own arena and pass a view in; the FFN module borrows it for
/// the duration of one forward pass.

#include <cstddef>

namespace hikoboshi::modules::ffn::detail {

/// Caller-owned workspace view for one `ffn_layer` forward call.
///
/// `intermediate_buffer` must hold at least `rows * intermediate_dim` floats
/// (the shape of the post-`w_in` activation output the template body writes
/// into). `intermediate_capacity` records the available float count so the
/// architecture can size its arena once at preparation time and reuse the
/// same view across layers.
struct FfnLayerWorkspace {
  float* intermediate_buffer;
  std::size_t intermediate_capacity;
};

}  // namespace hikoboshi::modules::ffn::detail

#endif  // HIKOBOSHI_MODULES_FFN_DETAIL_WORKSPACE_HPP
