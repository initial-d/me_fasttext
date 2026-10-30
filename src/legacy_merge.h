#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "real.h"
#include "row_merge.h"
#include "trie/datrie.h"

namespace fasttext {
using std::map;
using std::string;

// Shared by the legacy exporter and the optional native structural planner.
// The legacy defaults remain unchanged. Exact mode compares float bytes,
// including signed zero / NaN payloads, rather than a rounded cosine of 1.
struct RowSharingPolicy {
    bool exact = false;
    double cosine = 0.99;
    double norm_ratio = 0.99;
    bool accepts(const float* a, const float* b, size_t dim) const {
        return exact ? std::memcmp(a, b, dim * sizeof(float)) == 0
                     : joint_row_similarity(a, b, dim, cosine, norm_ratio);
    }
};

struct LegacyMergeState {
    string pq;
    size_t pid;
    float* cur;
    DATrie* pngramdat;
    int32_t dim;
    bool initialized = false;
    map<size_t, string>* original_keys = nullptr;
    std::vector<std::pair<string, size_t>> ancestors;
    RowSharingPolicy policy;
};

inline string reverse(string word) {
    string str;
    std::string ngram;
    if(word.size() == 0) return str;
    for (size_t j = 0; j < word.size();) {
        ngram.push_back(word[j++]);
        while (j < word.size() && (word[j] & 0xC0) == 0x80) {
            ngram.push_back(word[j++]);
        }
        str = ngram + str;
        ngram = "";
    }
    return str;
}

inline int datrie_foreach_cal(const char* key, size_t value, void* udata) {
    LegacyMergeState* state = static_cast<LegacyMergeState*>(udata);
    real* pos = state->cur;
    if (state->original_keys) (*state->original_keys)[value] = key;
    const string current(key);
    // Keep the visited prefix ancestry across sibling branches. Remembering
    // only the preceding key loses a valid shorter ancestor after a mismatch.
    while (!state->ancestors.empty() &&
           current.compare(0, state->ancestors.back().first.size(),
                           state->ancestors.back().first) != 0) {
        state->ancestors.pop_back();
    }
    size_t representative = value;
    for (auto entry = state->ancestors.rbegin(); entry != state->ancestors.rend(); ++entry) {
        if (state->policy.accepts(pos + entry->second * state->dim,
                                  pos + value * state->dim, state->dim)) {
            representative = entry->second;
            break;
        }
    }
    datrie_insert_key(state->pngramdat, reverse(current).c_str(), representative);
    state->ancestors.emplace_back(current, representative);
    state->pq = current;
    state->pid = representative;
    state->initialized = true;
    return 1;
}

// Prefix then suffix sharing can form a chain across passes. Verify every
// original key/row directly against its final representative before compacting.
struct FinalMergeGuard {
    real* rows;
    int32_t dim;
    DATrie* merged;
    const map<size_t, string>* original_keys;
    RowSharingPolicy policy;
    FinalMergeGuard(real* r, int32_t d, DATrie* m, const map<size_t, string>* k,
                    RowSharingPolicy p = RowSharingPolicy())
        : rows(r), dim(d), merged(m), original_keys(k), policy(p) {}
};

inline int datrie_foreach_guard(const char* key, size_t value, void* udata) {
    FinalMergeGuard* state = static_cast<FinalMergeGuard*>(udata);
    size_t representative;
    if (datrie_search_exact_match(state->merged, key, &representative) == -1) {
        throw std::runtime_error("missing ngram after merge");
    }
    const string& rep_key = state->original_keys->at(representative);
    if (!structurally_related(key, rep_key) ||
        !state->policy.accepts(state->rows + value * state->dim,
                               state->rows + representative * state->dim, state->dim)) {
        datrie_insert_key(state->merged, key, value);
    }
    return 1;
}

}  // namespace fasttext
