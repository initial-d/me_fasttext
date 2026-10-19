/* POSIX mmap and zlib; independent of the legacy FastText/trie libraries. */
#include "meft_rows.h"
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

/* MEFTNAT1 + six little-endian uint64 fields:
 * version (1 or 2), nrows, rowbytes, unique_rows, block_rows, nblocks.
 * Followed by nrows uint64 forwarding IDs and nblocks descriptors:
 * absolute offset, stored bytes, raw bytes, codec (0=raw, 1=zlib,
 * 2=zlib over previous-row XOR residuals, version 2 only).
 */
#define HEADER 56u
#define DESC 32u
struct meft_rows {
    unsigned char *data;
    size_t size;
    uint64_t nrows, width, unique, block_rows, nblocks, desc_start;
    unsigned char *scratch;
    size_t scratch_capacity;
    uint64_t last_blocks, last_decodes;
};
struct request { uint64_t block, within; size_t position; };
static uint64_t le64(const unsigned char *p) {
    uint64_t x = 0;
    for (unsigned i = 0; i < 8; ++i) x |= (uint64_t)p[i] << (8 * i);
    return x;
}
static int fail(char *out, size_t cap, const char *msg) {
    if (out && cap) snprintf(out, cap, "%s", msg);
    return -1;
}
static int span(uint64_t start, uint64_t count, uint64_t total) {
    return start <= total && count <= total - start;
}
void meft_rows_close(meft_rows *s) {
    if (!s) return;
    if (s->data) munmap(s->data, s->size);
    free(s->scratch);
    free(s);
}
meft_rows *meft_rows_open(const char *path, char *err, size_t cap) {
    struct stat st;
    int fd = path ? open(path, O_RDONLY) : -1;
    if (fd < 0) { fail(err, cap, "cannot open row file"); return NULL; }
    if (fstat(fd, &st) || st.st_size < HEADER || (uintmax_t)st.st_size > SIZE_MAX) {
        close(fd); fail(err, cap, "invalid row file size"); return NULL;
    }
    meft_rows *s = calloc(1, sizeof(*s));
    if (!s) { close(fd); fail(err, cap, "allocation failed"); return NULL; }
    s->size = (size_t)st.st_size;
    void *mapping = mmap(NULL, s->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) { free(s); fail(err, cap, "mmap failed"); return NULL; }
    s->data = mapping;
    const char *reason = "invalid native row header";
    uint64_t version = le64(s->data + 8);
    if (memcmp(s->data, "MEFTNAT1", 8) || (version != 1 && version != 2)) goto invalid;
    s->nrows = le64(s->data + 16); s->width = le64(s->data + 24);
    s->unique = le64(s->data + 32); s->block_rows = le64(s->data + 40);
    s->nblocks = le64(s->data + 48);
    if (!s->width || !s->block_rows || s->unique > s->nrows ||
        s->nrows > (s->size - HEADER) / 8 ||
        (s->nrows && !s->unique)) goto invalid;
    s->desc_start = HEADER + s->nrows * 8;
    if (s->nblocks != s->unique / s->block_rows + (s->unique % s->block_rows != 0) ||
        s->nblocks > (s->size - s->desc_start) / DESC) goto invalid;
    uint64_t previous_end = s->desc_start + s->nblocks * DESC;
    for (uint64_t b = 0; b < s->nblocks; ++b) {
        const unsigned char *d = s->data + s->desc_start + b * DESC;
        uint64_t off = le64(d), stored = le64(d + 8), raw = le64(d + 16), codec = le64(d + 24);
        uint64_t rows = s->unique - b * s->block_rows;
        if (rows > s->block_rows) rows = s->block_rows;
        reason = "invalid block descriptor";
        if (rows > UINT64_MAX / s->width || raw != rows * s->width ||
            raw > SIZE_MAX || raw > ULONG_MAX || stored > ULONG_MAX ||
            off != previous_end || !span(off, stored, s->size) || !stored ||
            codec > (version == 1 ? 1u : 2u) ||
            (!codec && stored != raw)) goto invalid;
        previous_end = off + stored;
    }
    reason = "unexpected trailing or missing bytes";
    if (previous_end != s->size) goto invalid;
    /* Map bounds are checked at open; individual row IDs are validated on gather.
     * No full payload/forwarding-table scan is needed for lazy startup. */
    return s;
invalid:
    meft_rows_close(s); fail(err, cap, reason); return NULL;
}
uint64_t meft_rows_count(const meft_rows *s) { return s ? s->nrows : 0; }
uint64_t meft_rows_width(const meft_rows *s) { return s ? s->width : 0; }
uint64_t meft_rows_last_block_reads(const meft_rows *s) { return s ? s->last_blocks : 0; }
uint64_t meft_rows_last_decompressions(const meft_rows *s) { return s ? s->last_decodes : 0; }
static int order(const void *a, const void *b) {
    const struct request *x = a, *y = b;
    return x->block < y->block ? -1 : x->block > y->block;
}
int meft_rows_gather(meft_rows *s, const uint64_t *ids, size_t count,
                     void *out, size_t outbytes, char *err, size_t cap) {
    if (!s) return fail(err, cap, "null store");
    s->last_blocks = s->last_decodes = 0;
    if (count > SIZE_MAX / s->width || count * s->width > outbytes ||
        count > SIZE_MAX / sizeof(struct request) || (count && (!ids || !out)))
        return fail(err, cap, "invalid output capacity or query size");
    if (!count) return 0;
    struct request *requests = malloc(count * sizeof(*requests));
    if (!requests) return fail(err, cap, "allocation failed");
    const char *reason = NULL;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] >= s->nrows) { reason = "row ID out of bounds"; goto done; }
        uint64_t row = le64(s->data + HEADER + ids[i] * 8);
        if (row >= s->unique) { reason = "invalid forwarding ID"; goto done; }
        requests[i].block = row / s->block_rows;
        requests[i].within = row % s->block_rows;
        requests[i].position = i;
    }
    qsort(requests, count, sizeof(*requests), order);
    for (size_t i = 0; i < count;) {
        uint64_t block = requests[i].block;
        const unsigned char *d = s->data + s->desc_start + block * DESC;
        uint64_t off = le64(d), stored = le64(d + 8), raw = le64(d + 16), codec = le64(d + 24);
        const unsigned char *bytes = s->data + off;
        ++s->last_blocks;
        if (codec) {
            if (s->scratch_capacity < raw) {
                void *next = realloc(s->scratch, (size_t)raw);
                if (!next) { reason = "allocation failed"; goto done; }
                s->scratch = next; s->scratch_capacity = (size_t)raw;
            }
            uLongf actual = (uLongf)raw;
            if (uncompress(s->scratch, &actual, bytes, (uLong)stored) != Z_OK || actual != raw) {
                reason = "block decompression failed"; goto done;
            }
            /* Prediction resets at each block's first row. Bytewise XOR
             * reconstructs original bits without floating-point arithmetic. */
            if (codec == 2) {
                size_t width = (size_t)s->width;
                for (size_t start = width; start < (size_t)raw; start += width) {
                    unsigned char *current = s->scratch + start;
                    const unsigned char *previous = current - width;
                    size_t j = 0;
                    /* memcpy permits unaligned rows/odd widths without aliasing
                     * or alignment UB, and compilers lower it to word loads. */
                    for (; width - j >= sizeof(uint64_t); j += sizeof(uint64_t)) {
                        uint64_t a, b;
                        memcpy(&a, current + j, sizeof(a));
                        memcpy(&b, previous + j, sizeof(b));
                        a ^= b;
                        memcpy(current + j, &a, sizeof(a));
                    }
                    for (; j < width; ++j) current[j] ^= previous[j];
                }
            }
            bytes = s->scratch; ++s->last_decodes;
        }
        do {
            memcpy((unsigned char *)out + requests[i].position * s->width,
                   bytes + requests[i].within * s->width, (size_t)s->width);
            ++i;
        } while (i < count && requests[i].block == block);
    }
done:
    free(requests);
    return reason ? fail(err, cap, reason) : 0;
}
