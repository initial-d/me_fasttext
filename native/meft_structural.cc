#include "meft_structural.h"
#include "legacy_merge.h"
#include "trie/dastrie.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>

namespace {
struct Plan {
    DASTrie* index = nullptr;
    std::vector<std::pair<std::string, uint64_t>> flat;
    std::vector<uint64_t> representatives, order;
    bool built = false;
    ~Plan() { if (index) dastrie_delete_unsafe(index); }
};
struct DeleteDATrie { void operator()(DATrie* t) const { if (t) datrie_delete_unsafe(t); } };
using Trie = std::unique_ptr<DATrie, DeleteDATrie>;
void fail(char* error, size_t capacity, const char* message) {
    if (error && capacity) std::snprintf(error, capacity, "%s", message);
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void write_u64(std::ostream& out, uint64_t value) {
    char bytes[8];
    for (size_t i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 255);
    out.write(bytes, 8);
}
}

extern "C" void* meft_structural_build(const char* const* keys, const uint64_t* ids,
    size_t count, const float* rows, size_t nrows, size_t dim, int mode,
    double cosine, double norm_ratio, char* error, size_t error_size) {
    try {
        require(rows && nrows && dim && dim <= INT32_MAX && nrows <= SIZE_MAX / dim / sizeof(float),
                "invalid float32 row dimensions");
        require((!count || (keys && ids)) && count <= nrows, "invalid keys or IDs");
        require(mode >= 0 && mode <= 2 && cosine > 0 && cosine <= 1 && norm_ratio > 0 && norm_ratio <= 1,
                "invalid sharing policy");
        std::unique_ptr<Plan> p(new Plan);
        Trie source(datrie_new()), reversed(datrie_new()), merged(datrie_new());
        require(source && reversed && merged, "cannot allocate DATrie");
        std::set<uint64_t> used_ids;
        for (size_t i = 0; i < count; ++i) {
            require(keys[i] && *keys[i] && ids[i] < nrows, "invalid key or original ID");
            require(used_ids.insert(ids[i]).second, "duplicate key-bound original ID");
            p->flat.emplace_back(keys[i], ids[i]);
        }
        std::sort(p->flat.begin(), p->flat.end());
        for (size_t i = 0; i < count; ++i) {
            require(!i || p->flat[i-1].first != p->flat[i].first, "duplicate key");
            datrie_insert_key(source.get(), p->flat[i].first.c_str(), p->flat[i].second);
        }
        p->index = dastrie_new();
        require(p->index, "cannot allocate DASTrie");
        dastrie_build_convert(p->index, source.get());
        for (const auto& entry : p->flat) {
            size_t id = 0;
            require(dastrie_search_exact_match(p->index, entry.first.c_str(), &id) != -1 && id == entry.second,
                    "DASTrie lost original ID");
        }
        p->representatives.resize(nrows);
        std::iota(p->representatives.begin(), p->representatives.end(), uint64_t(0));
        if (mode) {
            fasttext::RowSharingPolicy policy;
            policy.exact = mode == 1; policy.cosine = cosine; policy.norm_ratio = norm_ratio;
            std::map<size_t, std::string> original_keys;
            fasttext::LegacyMergeState prefix;
            prefix.cur = const_cast<float*>(rows); prefix.dim = static_cast<int32_t>(dim);
            prefix.pngramdat = reversed.get(); prefix.original_keys = &original_keys; prefix.policy = policy;
            datrie_foreach_keys(source.get(), fasttext::datrie_foreach_cal, &prefix);
            fasttext::LegacyMergeState suffix;
            suffix.cur = const_cast<float*>(rows); suffix.dim = static_cast<int32_t>(dim);
            suffix.pngramdat = merged.get(); suffix.policy = policy;
            datrie_foreach_keys(reversed.get(), fasttext::datrie_foreach_cal, &suffix);
            fasttext::FinalMergeGuard guard{const_cast<float*>(rows), static_cast<int32_t>(dim),
                                          merged.get(), &original_keys, policy};
            datrie_foreach_keys(source.get(), fasttext::datrie_foreach_guard, &guard);
            for (const auto& entry : p->flat) {
                size_t representative = nrows;
                require(datrie_search_exact_match(merged.get(), entry.first.c_str(), &representative) != -1 &&
                        representative < nrows && used_ids.count(representative), "invalid representative");
                p->representatives[entry.second] = representative;
            }
            // Resolve to roots before dense remapping. A final direct check
            // prevents prefix/suffix chains from bypassing either criterion.
            const uint64_t unset = UINT64_MAX;
            std::vector<uint64_t> roots(nrows, unset);
            std::vector<unsigned char> active(nrows, 0);
            for (size_t i = 0; i < nrows; ++i) {
                if (roots[i] != unset) continue;
                std::vector<uint64_t> path;
                uint64_t current = i;
                while (roots[current] == unset && !active[current]) {
                    active[current] = 1; path.push_back(current);
                    if (p->representatives[current] == current) { roots[current] = current; break; }
                    current = p->representatives[current];
                }
                if (roots[current] == unset) {
                    // Conservatively restore every visited row in a cycle.
                    for (uint64_t id : path) roots[id] = id;
                } else {
                    for (uint64_t id : path) roots[id] = roots[current];
                }
                for (uint64_t id : path) active[id] = 0;
            }
            for (const auto& entry : p->flat) {
                uint64_t id = entry.second, root = roots[id];
                if (root != id && (!fasttext::structurally_related(entry.first, original_keys.at(root)) ||
                    !policy.accepts(rows + id * dim, rows + root * dim, dim))) roots[id] = id;
            }
            p->representatives.swap(roots);
        }
        // Unindexed special/hash rows first; marked keyed rows in trie order.
        // Dead IDs follow solely to provide pack_native's full permutation.
        for (size_t i = 0; i < nrows; ++i) if (!used_ids.count(i)) p->order.push_back(i);
        for (const auto& entry : p->flat)
            if (p->representatives[entry.second] == entry.second) p->order.push_back(entry.second);
        for (size_t i = 0; i < nrows; ++i)
            if (p->representatives[i] != i) p->order.push_back(i);
        require(p->order.size() == nrows, "invalid survivor order");
        for (size_t i = 0; i < nrows; ++i)
            require(p->representatives[p->representatives[i]] == p->representatives[i], "non-idempotent plan");
        p->built = true;
        return p.release();
    } catch (const std::exception& e) { fail(error, error_size, e.what()); return nullptr; }
    catch (...) { fail(error, error_size, "structural build failed"); return nullptr; }
}

extern "C" void meft_structural_close(void* plan) { delete static_cast<Plan*>(plan); }
extern "C" int meft_structural_maps(void* plan, uint64_t* representatives, uint64_t* order,
    size_t count, char* error, size_t error_size) {
    Plan* p = static_cast<Plan*>(plan);
    if (!p || !p->built || !representatives || !order || count != p->representatives.size()) {
        fail(error, error_size, "invalid map buffer or plan"); return -1;
    }
    std::copy(p->representatives.begin(), p->representatives.end(), representatives);
    std::copy(p->order.begin(), p->order.end(), order);
    return 0;
}
extern "C" int meft_structural_lookup(void* plan, const char* const* keys, size_t count,
    uint64_t* ids, int index, char* error, size_t error_size) {
    try {
        Plan* p = static_cast<Plan*>(plan);
        require(p && (!count || (keys && ids)) && (index == 0 || index == 1) &&
                (index || p->built), "invalid lookup arguments");
        for (size_t i = 0; i < count; ++i) {
            require(keys[i], "null query key");
            ids[i] = UINT64_MAX;
            if (index) {
                size_t id = 0;
                if (dastrie_search_exact_match(p->index, keys[i], &id) != -1) ids[i] = id;
            } else {
                const auto it = std::lower_bound(p->flat.begin(), p->flat.end(), keys[i],
                    [](const std::pair<std::string, uint64_t>& entry, const char* key) { return entry.first < key; });
                if (it != p->flat.end() && it->first == keys[i]) ids[i] = it->second;
            }
        }
        return 0;
    } catch (const std::exception& e) { fail(error, error_size, e.what()); return -1; }
    catch (...) { fail(error, error_size, "structural lookup failed"); return -1; }
}
extern "C" int meft_structural_save_index(void* plan, const char* path, int index,
    char* error, size_t error_size) {
    try {
        Plan* p = static_cast<Plan*>(plan);
        require(p && path && (index == 0 || index == 1) && (index || p->built), "invalid index output");
        if (index) {
            require(dastrie_store_to_file(p->index, path) > 0, "cannot save DASTrie");
        } else {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write("MEFTKEY1", 8); write_u64(out, p->flat.size());
            uint64_t offset = 0;
            for (const auto& entry : p->flat) { write_u64(out, offset); offset += entry.first.size(); }
            write_u64(out, offset);
            for (const auto& entry : p->flat) write_u64(out, entry.second);
            for (const auto& entry : p->flat) out.write(entry.first.data(), entry.first.size());
            out.close(); require(static_cast<bool>(out), "cannot save flat index");
        }
        return 0;
    } catch (const std::exception& e) { fail(error, error_size, e.what()); return -1; }
    catch (...) { fail(error, error_size, "index save failed"); return -1; }
}
extern "C" void* meft_structural_open_index(const char* path, char* error, size_t error_size) {
    try {
        require(path, "null index path");
        std::unique_ptr<Plan> p(new Plan);
        p->index = dastrie_load_from_file(path);
        require(p->index, "cannot load trusted DASTrie index");
        return p.release();
    } catch (const std::exception& e) { fail(error, error_size, e.what()); return nullptr; }
    catch (...) { fail(error, error_size, "index load failed"); return nullptr; }
}
