#include <hikoboshi/primitives/alignment/smith_waterman.hpp>
#include <hikoboshi/primitives/alignment/traceback.hpp>
#include <hikoboshi/primitives/compute/gather.hpp>
#include <hikoboshi/primitives/compute/knn.hpp>
#include <hikoboshi/primitives/compute/layer_norm.hpp>
#include <hikoboshi/primitives/compute/log_softmax.hpp>
#include <hikoboshi/primitives/compute/rbf.hpp>
#include <hikoboshi/primitives/compute/reduce.hpp>
#include <hikoboshi/primitives/compute/softmax.hpp>
#include <hikoboshi/primitives/linalg/gemm.hpp>

namespace hikoboshi::primitives {

// Translation unit anchor for the scalar primitives library; pulls every
// chartered kernel header so missing prototypes fail at this layer rather than
// at the dispatch seam.
void primitives_layer_anchor() noexcept {}

}  // namespace hikoboshi::primitives
