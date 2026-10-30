#ifndef MEFT_STRUCTURAL_H
#define MEFT_STRUCTURAL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Optional Linux x86-64 bridge to the original bundled DATrie libraries.
 * Keys are unique nonempty NUL-terminated UTF-8 strings with unique original
 * row IDs. Unindexed rows retain their IDs. modes: 0=layout, 1=byte-exact
 * structural sharing, 2=approximate cosine AND norm-ratio structural sharing.
 * float32 rows are borrowed only for build; plans own their index and maps.
 * All errors are returned through the caller's buffer; no C++ exceptions escape.
 * Handles are single-worker. Existing numeric/hash queries need no key index.
 */
void* meft_structural_build(const char* const* keys, const uint64_t* ids,
    size_t count, const float* rows, size_t nrows, size_t dim, int mode,
    double cosine, double norm_ratio, char* error, size_t error_size);
void meft_structural_close(void* plan);
int meft_structural_maps(void* plan, uint64_t* representatives, uint64_t* order,
    size_t count, char* error, size_t error_size);
/* Both lookup implementations return original numeric IDs, including after
 * compaction. index=0 uses sorted keys + binary search, index=1 DASTrie.
 * index=2 adds an owned std::unordered_map baseline on built plans only.
 * Missing keys return UINT64_MAX. Flat/hash lookup is available on built plans.
 */
int meft_structural_lookup(void* plan, const char* const* keys, size_t count,
    uint64_t* ids, int index, char* error, size_t error_size);
int meft_structural_save_index(void* plan, const char* path, int index,
    char* error, size_t error_size);
/* Only trusted, self-produced DASTrie files: legacy serialization is not an
 * untrusted-file format and depends on the bundled library / architecture.
 */
void* meft_structural_open_index(const char* path, char* error, size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
