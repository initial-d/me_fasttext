#include "meft_structural.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

int main() {
    char error[256] = {};
    const char* keys[] = {"a", "ab", "abc", "ax", "ay", "记", "记忆"};
    uint64_t ids[] = {1, 2, 3, 4, 5, 6, 7};
    float rows[] = {0, 0, 1, 0, .996f, .089f, .984f, .178f, -1, 0, 2, 0, 0, 1, 0, 1};
    for (int mode = 0; mode <= 2; ++mode) {
        for (size_t repeat = 0; repeat < 20; ++repeat) {
            void* plan = meft_structural_build(keys, ids, 7, rows, 8, 2, mode, .99, .99, error, sizeof(error));
            assert(plan);
            uint64_t reps[8], order[8], found[8];
            assert(!meft_structural_maps(plan, reps, order, 8, error, sizeof(error)));
            assert(reps[0] == 0 && reps[3] == 3 && reps[4] == 4 && reps[5] == 5);
            assert(reps[2] == (mode == 2 ? 1u : 2u));
            assert(reps[7] == (mode ? 6u : 7u));
            bool seen[8] = {};
            for (size_t i = 0; i < 8; ++i) {
                assert(reps[reps[i]] == reps[i]);
                assert(order[i] < 8 && !seen[order[i]]); seen[order[i]] = true;
            }
            const char* queries[] = {"记忆", "ab", "abc", "a", "unknown", "", "ax", "ay"};
            for (int index = 0; index <= 2; ++index) {
                assert(!meft_structural_lookup(plan, queries, 8, found, index, error, sizeof(error)));
                assert(found[0] == 7 && found[1] == 2 && found[2] == 3 && found[3] == 1);
                assert(found[4] == UINT64_MAX && found[5] == UINT64_MAX && found[6] == 4 && found[7] == 5);
            }
            assert(meft_structural_maps(plan, reps, order, 7, error, sizeof(error)));
            assert(meft_structural_lookup(plan, queries, 8, nullptr, 1, error, sizeof(error)));
            assert(meft_structural_lookup(plan, queries, 8, found, 3, error, sizeof(error)));
            assert(!meft_structural_lookup(plan, nullptr, 0, nullptr, 1, error, sizeof(error)));
            meft_structural_close(plan);
        }
    }
    assert(!meft_structural_build(nullptr, nullptr, 7, rows, 8, 2, 1, .99, .99, error, sizeof(error)));
    assert(!meft_structural_build(keys, ids, 7, rows, 8, 2, 2, NAN, .99, error, sizeof(error)));
    uint64_t duplicated_ids[] = {1, 1, 3, 4, 5, 6, 7};
    assert(!meft_structural_build(keys, duplicated_ids, 7, rows, 8, 2, 2, .99, .99, error, sizeof(error)));
    meft_structural_close(nullptr);
}
