#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace fasttext {

// Whole-key ancestry, not merely a shared first/last character. UTF-8 keys
// are compared as bytes; valid complete keys keep code-point boundaries.
inline bool prefix_related(const std::string& a, const std::string& b) {
  if (a.empty() || b.empty()) return false;
  const std::string& shorter = a.size() <= b.size() ? a : b;
  const std::string& longer = a.size() <= b.size() ? b : a;
  return longer.compare(0, shorter.size(), shorter) == 0;
}

inline bool structurally_related(const std::string& a, const std::string& b) {
  if (prefix_related(a, b)) return true;
  if (a.empty() || b.empty()) return false;
  const std::string& shorter = a.size() <= b.size() ? a : b;
  const std::string& longer = a.size() <= b.size() ? b : a;
  return longer.compare(longer.size() - shorter.size(), shorter.size(), shorter) == 0;
}

// Approximate sharing: both direction and magnitude must pass. This does
// not bound downstream task quality and is not byte-exact compression.
inline bool joint_row_similarity(const float* a, const float* b, std::size_t dim,
                                 double cosine_threshold = 0.99,
                                 double ratio_threshold = 0.99) {
  if (dim == 0) return false;
  double aa = 0, bb = 0, ab = 0;
  bool identical = true;
  for (std::size_t j = 0; j < dim; ++j) {
    if (!std::isfinite(a[j]) || !std::isfinite(b[j])) return false;
    identical = identical && a[j] == b[j];
    aa += static_cast<double>(a[j]) * a[j];
    bb += static_cast<double>(b[j]) * b[j];
    ab += static_cast<double>(a[j]) * b[j];
  }
  if (aa == 0 || bb == 0) return identical;  // two zero vectors only
  const double ratio = std::sqrt(std::min(aa, bb) / std::max(aa, bb));
  const double cosine = ab / (std::sqrt(aa) * std::sqrt(bb));
  return cosine >= cosine_threshold && ratio >= ratio_threshold;
}

}  // namespace fasttext
