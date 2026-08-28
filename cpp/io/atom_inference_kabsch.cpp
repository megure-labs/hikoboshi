#include <hikoboshi/io/atom_inference.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <hikoboshi/universal/structure.hpp>

namespace hikoboshi::io {
namespace {

constexpr std::size_t kAxisCount = universal::kCoordinateAxisCount;
constexpr std::size_t kMaxTemplateAtoms = 25;

std::size_t coord_offset(std::size_t atom, std::size_t axis) noexcept {
  return atom * kAxisCount + axis;
}

// Compute the 3x3 covariance matrix H = sum_i p_i^T * q_i for paired centered
// points. Returns the centroid offsets so we can recover translations later.
void compute_covariance(const double p[][3], const double q[][3], int n,
                        double centroid_p[3], double centroid_q[3],
                        double h[3][3]) noexcept {
  centroid_p[0] = centroid_p[1] = centroid_p[2] = 0.0;
  centroid_q[0] = centroid_q[1] = centroid_q[2] = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 3; ++j) {
      centroid_p[j] += p[i][j];
      centroid_q[j] += q[i][j];
    }
  }
  for (int j = 0; j < 3; ++j) {
    centroid_p[j] /= n;
    centroid_q[j] /= n;
  }

  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      h[a][b] = 0.0;
    }
  }
  for (int i = 0; i < n; ++i) {
    const double pa[3] = {p[i][0] - centroid_p[0], p[i][1] - centroid_p[1],
                          p[i][2] - centroid_p[2]};
    const double qb[3] = {q[i][0] - centroid_q[0], q[i][1] - centroid_q[1],
                          q[i][2] - centroid_q[2]};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        h[a][b] += pa[a] * qb[b];
      }
    }
  }
}

void jacobi_eigen3(double a[3][3], double v[3][3], double d[3]) noexcept {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      v[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (int sweep = 0; sweep < 64; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < 3; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        off += std::fabs(a[p][q]);
      }
    }
    if (off < 1e-12) {
      break;
    }
    for (int p = 0; p < 3; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        if (std::fabs(apq) < 1e-15) {
          continue;
        }
        const double theta = (aqq - app) / (2.0 * apq);
        const double t = (theta >= 0.0)
                             ? 1.0 / (theta + std::sqrt(1.0 + theta * theta))
                             : 1.0 / (theta - std::sqrt(1.0 + theta * theta));
        const double cs = 1.0 / std::sqrt(1.0 + t * t);
        const double sn = t * cs;
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for (int r = 0; r < 3; ++r) {
          if (r == p || r == q) {
            continue;
          }
          const double arp = a[r][p];
          const double arq = a[r][q];
          a[r][p] = cs * arp - sn * arq;
          a[r][q] = sn * arp + cs * arq;
          a[p][r] = a[r][p];
          a[q][r] = a[r][q];
        }
        for (int r = 0; r < 3; ++r) {
          const double vrp = v[r][p];
          const double vrq = v[r][q];
          v[r][p] = cs * vrp - sn * vrq;
          v[r][q] = sn * vrp + cs * vrq;
        }
      }
    }
  }
  d[0] = a[0][0];
  d[1] = a[1][1];
  d[2] = a[2][2];
}

void matmul3(const double a[3][3],
             const double b[3][3],
             double out[3][3]) noexcept {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += a[i][k] * b[k][j];
      }
      out[i][j] = sum;
    }
  }
}

void transpose3(const double m[3][3], double out[3][3]) noexcept {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = m[j][i];
    }
  }
}

double det3(const double m[3][3]) noexcept {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

bool kabsch_local(const double p[][3],
                  const double q[][3],
                  int n,
                  double r[3][3],
                  double t[3],
                  double& rmsd) noexcept {
  if (n < 3) {
    return false;
  }
  double centroid_p[3];
  double centroid_q[3];
  double h[3][3];
  compute_covariance(p, q, n, centroid_p, centroid_q, h);

  double ht[3][3];
  transpose3(h, ht);
  double hth[3][3];
  matmul3(ht, h, hth);

  double evec[3][3];
  double eval[3];
  jacobi_eigen3(hth, evec, eval);

  int idx[3] = {0, 1, 2};
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      if (eval[idx[j]] > eval[idx[i]]) {
        std::swap(idx[i], idx[j]);
      }
    }
  }
  double sorted_eval[3];
  double v[3][3];
  for (int i = 0; i < 3; ++i) {
    sorted_eval[i] = eval[idx[i]];
    for (int j = 0; j < 3; ++j) {
      v[j][i] = evec[j][idx[i]];
    }
  }
  if (sorted_eval[0] <= 1e-12 || sorted_eval[1] <= 1e-12) {
    return false;
  }

  double sigma[3];
  sigma[0] = std::sqrt(std::max(0.0, sorted_eval[0]));
  sigma[1] = std::sqrt(std::max(0.0, sorted_eval[1]));
  sigma[2] = std::sqrt(std::max(0.0, sorted_eval[2]));

  double u[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 3; ++i) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) {
        sum += h[i][k] * v[k][j];
      }
      u[i][j] = sigma[j] > 1e-12 ? sum / sigma[j] : 0.0;
    }
  }
  u[0][2] = u[1][0] * u[2][1] - u[2][0] * u[1][1];
  u[1][2] = u[2][0] * u[0][1] - u[0][0] * u[2][1];
  u[2][2] = u[0][0] * u[1][1] - u[1][0] * u[0][1];

  double ut[3][3];
  transpose3(u, ut);
  double vut[3][3];
  matmul3(v, ut, vut);
  const double s = det3(vut) >= 0.0 ? 1.0 : -1.0;

  double v_corr[3][3];
  for (int i = 0; i < 3; ++i) {
    v_corr[i][0] = v[i][0];
    v_corr[i][1] = v[i][1];
    v_corr[i][2] = s * v[i][2];
  }
  matmul3(v_corr, ut, r);

  for (int i = 0; i < 3; ++i) {
    t[i] = centroid_q[i] - (r[i][0] * centroid_p[0] +
                            r[i][1] * centroid_p[1] +
                            r[i][2] * centroid_p[2]);
  }

  double err = 0.0;
  for (int i = 0; i < n; ++i) {
    double mapped[3];
    for (int a = 0; a < 3; ++a) {
      mapped[a] = r[a][0] * p[i][0] + r[a][1] * p[i][1] +
                  r[a][2] * p[i][2] + t[a];
    }
    for (int a = 0; a < 3; ++a) {
      const double diff = mapped[a] - q[i][a];
      err += diff * diff;
    }
  }
  rmsd = std::sqrt(err / static_cast<double>(n));
  return true;
}

}  // namespace

bool atom_inference_kabsch_use(
    const AtomInferenceKabschUseRequest& request,
    const AtomInferenceKabschUseOutput& output) noexcept {
  if (request.template_coordinates == nullptr || request.coordinates == nullptr ||
      request.atom_sources == nullptr || output.coordinates == nullptr ||
      output.atom_sources == nullptr || request.template_atom_count == 0 ||
      request.template_atom_count > kMaxTemplateAtoms) {
    return false;
  }

  int anchor_count = 0;
  double anchors_template[kMaxTemplateAtoms][3]{};
  double anchors_observed[kMaxTemplateAtoms][3]{};
  std::size_t missing_atoms[kMaxTemplateAtoms]{};
  std::size_t missing_count = 0;
  for (std::size_t atom = 0; atom < request.template_atom_count; ++atom) {
    const auto source = request.atom_sources[atom];
    if (source == universal::AtomSource::Observed) {
      for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
        anchors_template[anchor_count][axis] =
            request.template_coordinates[coord_offset(atom, axis)];
        anchors_observed[anchor_count][axis] =
            request.coordinates[coord_offset(atom, axis)];
      }
      ++anchor_count;
    } else if (source == universal::AtomSource::Missing) {
      missing_atoms[missing_count++] = atom;
    }
  }

  if (missing_count == 0 || anchor_count < 3) {
    return false;
  }

  double rotation[3][3];
  double translation[3];
  double rmsd = 0.0;
  if (!kabsch_local(anchors_template, anchors_observed, anchor_count, rotation,
                    translation, rmsd)) {
    return false;
  }
  if (rmsd > static_cast<double>(request.rmsd_guard)) {
    return false;
  }

  for (std::size_t index = 0; index < missing_count; ++index) {
    const std::size_t atom = missing_atoms[index];
    const double template_point[3] = {
        static_cast<double>(request.template_coordinates[coord_offset(atom, 0)]),
        static_cast<double>(request.template_coordinates[coord_offset(atom, 1)]),
        static_cast<double>(request.template_coordinates[coord_offset(atom, 2)]),
    };
    for (std::size_t axis = 0; axis < kAxisCount; ++axis) {
      const double mapped =
          rotation[axis][0] * template_point[0] +
          rotation[axis][1] * template_point[1] +
          rotation[axis][2] * template_point[2] + translation[axis];
      output.coordinates[coord_offset(atom, axis)] = static_cast<float>(mapped);
    }
    output.atom_sources[atom] = universal::AtomSource::Inferred;
  }

  return true;
}

}  // namespace hikoboshi::io
