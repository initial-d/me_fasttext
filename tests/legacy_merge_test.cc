#include "legacy_merge.h"

#include <cassert>
#include <limits>
#include <vector>

// Portable trie test double: execute the production callbacks without the
// platform-specific bundled archives. Full exporter syntax is checked in CI.
struct _DATrie { std::map<std::string, size_t> values; };
extern "C" int datrie_insert_key(DATrie* trie, const char* key, size_t value) {
  trie->values[key] = value;
  return 1;
}
extern "C" int datrie_search_exact_match(DATrie* trie, const char* key, size_t* out) {
  auto entry = trie->values.find(key);
  if (entry == trie->values.end()) return -1;
  *out = entry->second;
  return 1;
}

int main() {
  using namespace fasttext;
  const float a[] = {1, 0}, opposite[] = {-1, 0}, scaled[] = {2, 0};
  const float zero[] = {0, 0}, nan[] = {1, std::numeric_limits<float>::quiet_NaN()};
  assert(joint_row_similarity(a, a, 2));
  assert(!joint_row_similarity(a, opposite, 2));
  assert(!joint_row_similarity(a, scaled, 2));
  assert(joint_row_similarity(zero, zero, 2));
  assert(!joint_row_similarity(zero, a, 2));
  assert(!joint_row_similarity(nan, nan, 2));
  assert(structurally_related("ab", "abc"));
  assert(structurally_related("bc", "abc"));
  assert(!structurally_related("abc", "abd"));
  assert(!structurally_related("ab", "xyz"));
  assert(reverse("你好a") == "a好你");
  assert(prefix_related("你好", "你好啊"));

  for (int dim : {1, 2, 64, 128, 256}) {
    std::vector<float> rows(4 * dim, 0);
    rows[0] = 1; rows[dim] = 1; rows[2 * dim] = -1; rows[3 * dim] = 1;
    DATrie trie;
    std::map<size_t, std::string> keys;
    LegacyMergeState state;
    state.cur = rows.data(); state.dim = dim; state.pid = 0;
    state.pngramdat = &trie; state.original_keys = &keys;
    // First visited key need not have row ID zero. It must retain its ID.
    datrie_foreach_cal("ab", 1, &state);
    assert(trie.values.at("ba") == 1);
    datrie_foreach_cal("abc", 0, &state);
    assert(trie.values.at("cba") == 1);
    datrie_foreach_cal("abcd", 2, &state);
    assert(trie.values.at("dcba") == 2);
    datrie_foreach_cal("xyz", 3, &state);
    assert(trie.values.at("zyx") == 3);
  }

  // A mismatching sibling must not discard a shorter, still-valid ancestor.
  float branching_rows[] = {1, 0, -1, 0, 1, 0};
  DATrie branching;
  LegacyMergeState branch_state;
  branch_state.cur = branching_rows; branch_state.dim = 2;
  branch_state.pngramdat = &branching;
  datrie_foreach_cal("ab", 0, &branch_state);
  datrie_foreach_cal("abc", 1, &branch_state);
  datrie_foreach_cal("abd", 2, &branch_state);
  assert(branching.values.at("cba") == 1);
  assert(branching.values.at("dba") == 0);

  // Two individually acceptable 7-degree steps must not become a 14-degree
  // replacement after prefix + suffix passes. Check original -> final row.
  const double pi = std::acos(-1.0);
  float rows[] = {1, 0, float(std::cos(7*pi/180)), float(std::sin(7*pi/180)),
                  float(std::cos(14*pi/180)), float(std::sin(14*pi/180))};
  assert(joint_row_similarity(rows, rows+2, 2));
  assert(joint_row_similarity(rows+2, rows+4, 2));
  DATrie final_trie;
  final_trie.values = {{"ab", 0}, {"abc", 0}, {"abcd", 0}, {"xyz", 0}};
  std::map<size_t, std::string> keys = {{0, "ab"}, {1, "abc"}, {2, "abcd"}, {3, "xyz"}};
  FinalMergeGuard guard{rows, 2, &final_trie, &keys};
  datrie_foreach_guard("abcd", 2, &guard);
  assert(final_trie.values.at("abcd") == 2);
  datrie_foreach_guard("abc", 1, &guard);
  assert(final_trie.values.at("abc") == 0);
  // Structural rejection happens before touching the unrelated vector.
  datrie_foreach_guard("xyz", 3, &guard);
  assert(final_trie.values.at("xyz") == 3);
}
