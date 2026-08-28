#include <hikoboshi/primitives/alignment/smith_waterman.hpp>
#include <hikoboshi/primitives/alignment/soft_sw_numerics.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace hikoboshi::primitives::alignment {

namespace {

inline void zero_posteriors(float* posteriors, std::size_t cell_count) {
  for (std::size_t i = 0; i < cell_count; ++i) {
    posteriors[i] = 0.0F;
  }
}

inline void fill_neg_inf(float* workspace, std::size_t cell_count) {
  for (std::size_t i = 0; i < cell_count; ++i) {
    workspace[i] = kSoftSwNegInf;
  }
}

inline float clamp_log_posterior(float log_posterior) noexcept {
  return std::min(log_posterior, 0.0F);
}

inline float posterior_weight(float alpha, float log_z, float temperature) noexcept {
  if (alpha <= kSoftSwNegInf) {
    return 0.0F;
  }
  return safe_exp(clamp_log_posterior((alpha - log_z) / temperature));
}

// Softmax over four predecessors with -inf-safe handling. Mirrors the
// orihime reference: any term <= NINF contributes zero weight, the rest are
// normalized to sum to 1.
inline void softmax4_weights(float a, float b, float c, float d, float temperature,
                             float& wa, float& wb, float& wc, float& wd) noexcept {
  const float max_v = std::max({a, b, c, d});
  if (max_v <= kSoftSwNegInf) {
    wa = wb = wc = wd = 0.0F;
    return;
  }
  KahanSumFloat sum;
  if (a > kSoftSwNegInf) {
    wa = safe_exp((a - max_v) / temperature);
    sum.add(wa);
  } else {
    wa = 0.0F;
  }
  if (b > kSoftSwNegInf) {
    wb = safe_exp((b - max_v) / temperature);
    sum.add(wb);
  } else {
    wb = 0.0F;
  }
  if (c > kSoftSwNegInf) {
    wc = safe_exp((c - max_v) / temperature);
    sum.add(wc);
  } else {
    wc = 0.0F;
  }
  if (d > kSoftSwNegInf) {
    wd = safe_exp((d - max_v) / temperature);
    sum.add(wd);
  } else {
    wd = 0.0F;
  }
  const float total = sum.result();
  if (total > 0.0F) {
    const float inv_total = 1.0F / total;
    wa *= inv_total;
    wb *= inv_total;
    wc *= inv_total;
    wd *= inv_total;
  }
}

inline void softmax2_weights(float a, float b, float temperature,
                             float& wa, float& wb) noexcept {
  const float max_v = std::max(a, b);
  if (max_v <= kSoftSwNegInf) {
    wa = wb = 0.0F;
    return;
  }
  KahanSumFloat sum;
  if (a > kSoftSwNegInf) {
    wa = safe_exp((a - max_v) / temperature);
    sum.add(wa);
  } else {
    wa = 0.0F;
  }
  if (b > kSoftSwNegInf) {
    wb = safe_exp((b - max_v) / temperature);
    sum.add(wb);
  } else {
    wb = 0.0F;
  }
  const float total = sum.result();
  if (total > 0.0F) {
    const float inv_total = 1.0F / total;
    wa *= inv_total;
    wb *= inv_total;
  }
}

}  // namespace

void soft_smith_waterman_scalar(const SoftSmithWatermanScalarRequest& request,
                                SoftSmithWatermanScalarOutput& output) {
  const std::size_t lq = request.query_length;
  const std::size_t lt = request.target_length;
  const std::size_t score_cells = lq * lt;

  output.log_partition = kSoftSwNegInf;

  if (output.posteriors != nullptr && score_cells > 0) {
    zero_posteriors(output.posteriors, score_cells);
  }

  if (lq == 0 || lt == 0) {
    return;
  }

  const float temperature = request.temperature;
  if (!(temperature > 0.0F)) {
    return;
  }

  const std::size_t row_stride = lt + 1;
  const std::size_t needed_cells = (lq + 1) * row_stride;
  if (request.match_workspace == nullptr || request.insert_workspace == nullptr ||
      request.delete_workspace == nullptr ||
      request.match_grad_workspace == nullptr ||
      request.insert_grad_workspace == nullptr ||
      request.delete_grad_workspace == nullptr ||
      output.posteriors == nullptr ||
      request.workspace_cells < needed_cells) {
    return;
  }

  const float gap_open = request.gap_open;
  const float gap_extension = request.gap_extension;

  float* match_alpha = request.match_workspace;
  float* insert_alpha = request.insert_workspace;
  float* delete_alpha = request.delete_workspace;
  float* match_beta = request.match_grad_workspace;
  float* insert_beta = request.insert_grad_workspace;
  float* delete_beta = request.delete_grad_workspace;

  // Boundary: M[0,0] = 0 (the empty alignment), all other states/cells -inf.
  // The recurrence's explicit "0" inside logsumexp4 represents the local-SW
  // sky restart at every interior cell; forcing diagonal predecessor terms
  // to -inf when i==1 or j==1 keeps the empty-alignment cell from
  // double-counting that restart.
  fill_neg_inf(match_alpha, needed_cells);
  fill_neg_inf(insert_alpha, needed_cells);
  fill_neg_inf(delete_alpha, needed_cells);
  match_alpha[0] = 0.0F;

  // Forward pass.
  for (std::size_t i = 1; i <= lq; ++i) {
    const std::size_t row = i * row_stride;
    const std::size_t prev_row = (i - 1) * row_stride;
    for (std::size_t j = 1; j <= lt; ++j) {
      const float score = request.scores[(i - 1) * lt + (j - 1)];
      const std::size_t cell = row + j;
      const std::size_t cell_diag = prev_row + (j - 1);
      const std::size_t cell_up = prev_row + j;
      const std::size_t cell_left = row + (j - 1);

      const bool diag_valid = (i > 1 && j > 1);
      const float m_from_match =
          diag_valid ? match_alpha[cell_diag] + score : kSoftSwNegInf;
      const float m_from_insert =
          diag_valid ? insert_alpha[cell_diag] + score : kSoftSwNegInf;
      const float m_from_delete =
          diag_valid ? delete_alpha[cell_diag] + score : kSoftSwNegInf;
      const float m_sky = score;
      match_alpha[cell] =
          logsumexp4(m_from_match, m_from_insert, m_from_delete, m_sky, temperature);

      const float i_from_match =
          (i > 1) ? match_alpha[cell_up] + gap_open : kSoftSwNegInf;
      const float i_from_insert =
          (i > 1) ? insert_alpha[cell_up] + gap_extension : kSoftSwNegInf;
      insert_alpha[cell] = logsumexp2(i_from_match, i_from_insert, temperature);

      const float d_from_match =
          (j > 1) ? match_alpha[cell_left] + gap_open : kSoftSwNegInf;
      const float d_from_delete =
          (j > 1) ? delete_alpha[cell_left] + gap_extension : kSoftSwNegInf;
      delete_alpha[cell] = logsumexp2(d_from_match, d_from_delete, temperature);
    }
  }

  // Partition function: log Z = T * logsumexp_{i,j,state} alpha[state,i,j] / T.
  // Sum over the entire (lq+1) x (lt+1) grid including the boundary so the
  // empty-alignment cell M[0,0] = 0 contributes once.
  float max_alpha = kSoftSwNegInf;
  for (std::size_t idx = 0; idx < needed_cells; ++idx) {
    if (match_alpha[idx] > max_alpha) max_alpha = match_alpha[idx];
    if (insert_alpha[idx] > max_alpha) max_alpha = insert_alpha[idx];
    if (delete_alpha[idx] > max_alpha) max_alpha = delete_alpha[idx];
  }

  if (max_alpha <= kSoftSwNegInf) {
    return;
  }

  KahanSumFloat partition_sum;
  for (std::size_t idx = 0; idx < needed_cells; ++idx) {
    const float m_val = match_alpha[idx];
    if (m_val > kSoftSwNegInf) {
      partition_sum.add(safe_exp((m_val - max_alpha) / temperature));
    }
    const float i_val = insert_alpha[idx];
    if (i_val > kSoftSwNegInf) {
      partition_sum.add(safe_exp((i_val - max_alpha) / temperature));
    }
    const float d_val = delete_alpha[idx];
    if (d_val > kSoftSwNegInf) {
      partition_sum.add(safe_exp((d_val - max_alpha) / temperature));
    }
  }
  const float log_z = max_alpha + temperature * std::log(partition_sum.result());
  output.log_partition = log_z;

  // Backward pass: beta[state, i, j] starts as the cell's clamped contribution
  // exp((alpha - log_z)/T) and accumulates incoming contributions from
  // successors during the reverse traversal. Posteriors are read off the
  // total beta passing through M[i, j] for each (i, j).
  for (std::size_t idx = 0; idx < needed_cells; ++idx) {
    match_beta[idx] = posterior_weight(match_alpha[idx], log_z, temperature);
    insert_beta[idx] = posterior_weight(insert_alpha[idx], log_z, temperature);
    delete_beta[idx] = posterior_weight(delete_alpha[idx], log_z, temperature);
  }

  for (std::size_t ip1 = lq; ip1 >= 1; --ip1) {
    const std::size_t i = ip1;
    const std::size_t row = i * row_stride;
    const std::size_t prev_row = (i - 1) * row_stride;
    for (std::size_t jp1 = lt; jp1 >= 1; --jp1) {
      const std::size_t j = jp1;
      const float score = request.scores[(i - 1) * lt + (j - 1)];
      const std::size_t cell = row + j;
      const std::size_t cell_diag = prev_row + (j - 1);
      const std::size_t cell_up = prev_row + j;
      const std::size_t cell_left = row + (j - 1);

      const bool diag_valid = (i > 1 && j > 1);
      const float beta_m = match_beta[cell];
      if (beta_m != 0.0F) {
        const float m_from_match =
            diag_valid ? match_alpha[cell_diag] + score : kSoftSwNegInf;
        const float m_from_insert =
            diag_valid ? insert_alpha[cell_diag] + score : kSoftSwNegInf;
        const float m_from_delete =
            diag_valid ? delete_alpha[cell_diag] + score : kSoftSwNegInf;
        const float m_sky = score;

        float w_mm = 0.0F;
        float w_mi = 0.0F;
        float w_md = 0.0F;
        float w_msky = 0.0F;
        softmax4_weights(m_from_match, m_from_insert, m_from_delete, m_sky,
                         temperature, w_mm, w_mi, w_md, w_msky);

        // Posterior for scores[i-1, j-1]: beta_m times the sum of weights
        // for the four predecessor terms (which all carry the score with
        // derivative 1). The weights sum to 1 when at least one predecessor
        // is finite, so this collapses to beta_m, but writing it explicitly
        // keeps parity with the orihime kernel for fully-degenerate cells.
        output.posteriors[(i - 1) * lt + (j - 1)] +=
            beta_m * (w_mm + w_mi + w_md + w_msky);

        if (diag_valid) {
          if (w_mm > 0.0F) {
            match_beta[cell_diag] += beta_m * w_mm;
          }
          if (w_mi > 0.0F) {
            insert_beta[cell_diag] += beta_m * w_mi;
          }
          if (w_md > 0.0F) {
            delete_beta[cell_diag] += beta_m * w_md;
          }
        }
      }

      const float beta_i = insert_beta[cell];
      if (beta_i != 0.0F) {
        const float i_from_match =
            (i > 1) ? match_alpha[cell_up] + gap_open : kSoftSwNegInf;
        const float i_from_insert =
            (i > 1) ? insert_alpha[cell_up] + gap_extension : kSoftSwNegInf;

        float w_im = 0.0F;
        float w_ii = 0.0F;
        softmax2_weights(i_from_match, i_from_insert, temperature, w_im, w_ii);

        if (i > 1) {
          if (w_im > 0.0F) {
            match_beta[cell_up] += beta_i * w_im;
          }
          if (w_ii > 0.0F) {
            insert_beta[cell_up] += beta_i * w_ii;
          }
        }
      }

      const float beta_d = delete_beta[cell];
      if (beta_d != 0.0F) {
        const float d_from_match =
            (j > 1) ? match_alpha[cell_left] + gap_open : kSoftSwNegInf;
        const float d_from_delete =
            (j > 1) ? delete_alpha[cell_left] + gap_extension : kSoftSwNegInf;

        float w_dm = 0.0F;
        float w_dd = 0.0F;
        softmax2_weights(d_from_match, d_from_delete, temperature, w_dm, w_dd);

        if (j > 1) {
          if (w_dm > 0.0F) {
            match_beta[cell_left] += beta_d * w_dm;
          }
          if (w_dd > 0.0F) {
            delete_beta[cell_left] += beta_d * w_dd;
          }
        }
      }

      if (j == 1) {
        break;
      }
    }
    if (i == 1) {
      break;
    }
  }

  // Numerical drift can push posteriors slightly outside [0, 1]; clamp.
  for (std::size_t i = 0; i < lq; ++i) {
    for (std::size_t j = 0; j < lt; ++j) {
      const std::size_t idx = i * lt + j;
      float v = output.posteriors[idx];
      if (v < 0.0F) {
        v = 0.0F;
      } else if (v > 1.0F) {
        v = 1.0F;
      }
      output.posteriors[idx] = v;
    }
  }
}

}  // namespace hikoboshi::primitives::alignment
