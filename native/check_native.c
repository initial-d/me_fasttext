/* Sanitizer driver: exercises a valid artifact and mutations of its bytes. */
#include "meft_rows.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    char error[256];
    meft_rows *s = meft_rows_open(argv[1], error, sizeof(error));
    if (!s) return 0; /* malformed files are expected to be rejected */
    uint64_t n = meft_rows_count(s), width = meft_rows_width(s);
    if (width > 1048576) { meft_rows_close(s); return 0; }
    unsigned char *out = malloc((size_t)width * 4);
    if (!out) { meft_rows_close(s); return 1; }
    uint64_t ids[4] = {0, n ? n - 1 : 0, 0, n};
    (void)meft_rows_set_cache_bytes(s, 65536);
    (void)meft_rows_gather(s, ids, 3, out, (size_t)width * 4, error, sizeof(error));
    (void)meft_rows_gather(s, ids, 3, out, (size_t)width * 4, error, sizeof(error));
    (void)meft_rows_gather(s, ids, 4, out, (size_t)width * 4, error, sizeof(error));
    (void)meft_rows_set_cache_bytes(s, 1);
    (void)meft_rows_gather(s, ids, 3, out, (size_t)width * 4, error, sizeof(error));
    (void)meft_rows_set_cache_bytes(s, 0);
    uint64_t offsets[3] = {0, 2, 3}, bad_offsets[3] = {0, 3, 2};
    for (int dtype = 1; dtype <= 3; ++dtype) {
        (void)meft_rows_embedding_bag(s, ids, 3, offsets, 2, dtype, 1, UINT64_MAX, 2,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 4, offsets, 2, dtype, 0, UINT64_MAX, 1,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 3, bad_offsets, 2, dtype, 0, UINT64_MAX, 1,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 3, offsets, 2, dtype, 0, UINT64_MAX, 0,
                                     (float *)out, (size_t)width, error, sizeof(error));
        (void)meft_rows_embedding_bag(s, ids, 3, offsets, 2, dtype, 0, UINT64_MAX, 1,
                                     (float *)out, 0, error, sizeof(error));
    }
    free(out); meft_rows_close(s); return 0;
}
