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
/* output must contain count * width bytes and must not overlap ids.
 * Failure may leave partial output; ignore it unless return value is zero.
 * Reads are grouped by block and returned in original query order.
 */
int meft_rows_gather(meft_rows *store, const uint64_t *ids, size_t count,
                     void *output, size_t output_bytes, char *error, size_t error_capacity);
uint64_t meft_rows_last_block_reads(const meft_rows *store);
uint64_t meft_rows_last_decompressions(const meft_rows *store);
#ifdef __cplusplus
}
#endif
#endif
