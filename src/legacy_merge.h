#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#include "real.h"
#include "row_merge.h"
#include "trie/datrie.h"

namespace fasttext {
using std::map;
using std::string;

struct LegacyMergeState {
    string pq;
    size_t pid;
    float* cur;
    DATrie* pngramdat;
    int32_t dim;
    bool initialized = false;
    map<size_t, string>* original_keys = nullptr;
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
    real* pos = ((LegacyMergeState*)(udata))->cur;
    string pq = ((LegacyMergeState*)udata)->pq;
    size_t pid = ((LegacyMergeState*)udata)->pid;
    DATrie* pngramdat = ((LegacyMergeState*)udata)->pngramdat;
    
    LegacyMergeState* state = static_cast<LegacyMergeState*>(udata);
    if (state->original_keys) (*state->original_keys)[value] = key;
    bool merge = state->initialized && prefix_related(pq, key)
        && joint_row_similarity(pos + pid * state->dim,
                                pos + value * state->dim, state->dim);
    string rkey = reverse(key);

    if(merge) {
        datrie_insert_key(pngramdat, rkey.c_str(), pid);
        // Keep the representative key and row fixed; never compare only
        // against the preceding member of an approximate merge chain.
        ((LegacyMergeState*)udata)->pid = pid;
        //((LegacyMergeState*)udata)->pngramdat = pngramdat;
    } else {
        datrie_insert_key(pngramdat, rkey.c_str(), value);
        ((LegacyMergeState*)udata)->pq = key;
        ((LegacyMergeState*)udata)->pid = value;
        //((LegacyMergeState*)udata)->pngramdat = pngramdat;
    }
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
};

inline int datrie_foreach_guard(const char* key, size_t value, void* udata) {
    FinalMergeGuard* state = static_cast<FinalMergeGuard*>(udata);
    size_t representative;
    if (datrie_search_exact_match(state->merged, key, &representative) == -1) {
        throw std::runtime_error("missing ngram after merge");
    }
    const string& rep_key = state->original_keys->at(representative);
    if (!structurally_related(key, rep_key) ||
        !joint_row_similarity(state->rows + value * state->dim,
                              state->rows + representative * state->dim, state->dim)) {
        datrie_insert_key(state->merged, key, value);
    }
    return 1;
}

}  // namespace fasttext
