#ifndef MEFT_ROWS_H
#define MEFT_ROWS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Immutable CPU row storage. Each handle is single-threaded; use one per worker.
 * Rows are opaque bytes: caller owns dtype, tensor shape and head offsets.
 * All IDs are original global row IDs. Errors never cross the ABI as exceptions.
 * Reads MEFTNAT1 versions 1 and 2 (optional lossless XOR prediction).
 */
typedef struct meft_rows meft_rows;
meft_rows *meft_rows_open(const char *path, char *error, size_t error_capacity);
void meft_rows_close(meft_rows *store);
uint64_t meft_rows_count(const meft_rows *store);
uint64_t meft_rows_width(const meft_rows *store);
/* Optional decoded-block LRU cache, disabled at open. Setting a budget clears
 * all entries; payload bytes never exceed budget. At most 64 entries per handle.
 * Scratch and query locations are additional memory. Single-worker only. */
int meft_rows_set_cache_bytes(meft_rows *store, size_t budget);
uint64_t meft_rows_cache_bytes(const meft_rows *store);
/* output must contain count * width bytes and must not overlap ids.
 * Failure may leave partial output; ignore it unless return value is zero.
 * Reads are grouped by block and returned in original query order.
 */
int meft_rows_gather(meft_rows *store, const uint64_t *ids, size_t count,
                     void *output, size_t output_bytes, char *error, size_t error_capacity);
/* Gather in chunks of at most max_query_rows>0 requests, directly into the
 * caller's complete output. Bounds query-location/sort scratch, not output,
 * decoded blocks, cache or total RSS. Each chunk groups its own block reads;
 * crossing chunks can repeat decoding. Last-query counters aggregate all chunks.
 * Same original-ID order, no-overlapping-inputs and partial-error-output contract.
 */
int meft_rows_gather_chunked(meft_rows *store, const uint64_t *ids, size_t count,
                             void *output, size_t output_bytes, size_t max_query_rows,
                             char *error, size_t error_capacity);
/* Bounded gather + sum/mean for ragged bags. offsets contains bags+1 entries,
 * starts at 0 and ends at count. dtype: 1=LE float32, 2=LE IEEE float16,
 * 3=LE bfloat16. Output is bags * (width / element_bytes) native float32.
 * padding_id=UINT64_MAX disables padding; otherwise that row is excluded from
 * both sum and mean denominator. Empty bags return zero. max_chunk_rows>0
 * bounds staged rows. Reductions accumulate in float64 in original ID order;
 * stored bytes stay exact, but sums are not bit-identical to float32 reduction.
 * Same single-worker/no-overlapping-inputs/partial-error-output contract as gather.
 */
int meft_rows_embedding_bag(meft_rows *store, const uint64_t *ids, size_t count,
                           const uint64_t *offsets, size_t bags, int dtype, int mean,
                           uint64_t padding_id, size_t max_chunk_rows,
                           float *output, size_t output_floats, char *error, size_t error_capacity);
/* Weighted sum, with one LE scalar of the same dtype per query occurrence.
 * weights_bytes must cover count * element_bytes; repeated IDs retain distinct
 * weights. Padding excludes the row and its weight, even if nonfinite.
 * Products and sums use float64. No weight normalization or weighted mean.
 * Same capacities, staging bound and input/output ownership as the reducer above.
 * Existing unweighted ABI remains available unchanged.
 */
int meft_rows_embedding_bag_weighted(meft_rows *store, const uint64_t *ids, size_t count,
                                    const uint64_t *offsets, size_t bags, int dtype,
                                    const void *weights, size_t weights_bytes,
                                    uint64_t padding_id, size_t max_chunk_rows,
                                    float *output, size_t output_floats,
                                    char *error, size_t error_capacity);
uint64_t meft_rows_last_block_reads(const meft_rows *store);
uint64_t meft_rows_last_decompressions(const meft_rows *store);
uint64_t meft_rows_last_cache_hits(const meft_rows *store);
#ifdef __cplusplus
}
#endif
#endif
