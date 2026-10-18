/* Build: cc -std=c11 -Inative examples/native_gather.c native/meft_rows.c -lz -o /tmp/meft-gather */
#include "meft_rows.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <errno.h>
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s table.men row_id ...\n", argv[0]); return 1; }
    char error[256];
    meft_rows *s = meft_rows_open(argv[1], error, sizeof(error));
    if (!s) { fprintf(stderr, "%s\n", error); return 1; }
    size_t count = (size_t)argc - 2;
    uint64_t width = meft_rows_width(s);
    if (width > SIZE_MAX / count) { meft_rows_close(s); return 1; }
    uint64_t *ids = malloc(count * sizeof(*ids));
    unsigned char *out = malloc(count * (size_t)width);
    int status = 1;
    if (!ids || !out) goto done;
    for (size_t i = 0; i < count; ++i) {
        char *end;
        errno = 0;
        if (argv[i + 2][0] == '-') goto done;
        ids[i] = strtoull(argv[i + 2], &end, 10);
        if (errno || *end) goto done;
    }
    if (meft_rows_gather(s, ids, count, out, count * (size_t)width, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); goto done;
    }
    printf("rows=%zu width=%" PRIu64 " blocks=%" PRIu64 " decodes=%" PRIu64 "\n",
           count, width, meft_rows_last_block_reads(s), meft_rows_last_decompressions(s));
    for (size_t i = 0; i < count; ++i) {
        printf("id=%" PRIu64 " bytes=", ids[i]);
        for (uint64_t j = 0; j < width; ++j) printf("%02x", out[i * width + j]);
        puts("");
    }
    status = 0;
done:
    free(ids); free(out); meft_rows_close(s); return status;
}
