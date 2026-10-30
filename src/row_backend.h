#pragma once
#include <cstdint>
#include <vector>
#include "vector.h"

namespace fasttext {
/* Immutable serving rows. Implementations must preserve request order,
 * multiplicity and the legacy float32 accumulation, including chunk boundaries.
 * Single-worker handles; training continues to use Matrix/QMatrix. */
class RowBackend {
 public:
  virtual ~RowBackend() {}
  virtual void addRows(Vector&, const std::vector<int32_t>&) const = 0;
  virtual int64_t rows() const = 0;
  virtual int64_t width() const = 0;
};
}
