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
    free(out); meft_rows_close(s); return 0;
}
