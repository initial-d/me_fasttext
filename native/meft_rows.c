/* POSIX mmap and zlib; independent of the legacy FastText/trie libraries. */
#include "meft_rows.h"
#include <fcntl.h>
#include <float.h>
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
struct cached_block {
    uint64_t key, query; /* key is a block ID, or a forwarded row ID in row mode. */
    size_t size;
    struct cached_block *hash_next, *previous, *next;
    unsigned char bytes[];
};
struct meft_rows {
    unsigned char *data;
    size_t size;
    uint64_t nrows, width, unique, block_rows, nblocks, desc_start;
    unsigned char *scratch;
    size_t scratch_capacity;
    uint64_t last_blocks, last_decodes, last_hits, last_row_hits;
    int cache_rows;
    size_t cache_budget, cache_used, cache_count;
    size_t cache_buckets;
    uint64_t cache_query;
    struct cached_block **cache;
    struct cached_block *newest, *oldest;
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
    meft_rows_set_cache_bytes(s, 0);
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
uint64_t meft_rows_last_cache_hits(const meft_rows *s) { return s ? s->last_hits : 0; }
uint64_t meft_rows_last_row_cache_hits(const meft_rows *s) { return s ? s->last_row_hits : 0; }
uint64_t meft_rows_cache_bytes(const meft_rows *s) { return s ? s->cache_used : 0; }
int meft_rows_set_cache_bytes(meft_rows *s, size_t budget) {
    if (!s) return -1;
    while (s->newest) {
        struct cached_block *next = s->newest->next;
        free(s->newest);
        s->newest = next;
    }
    free(s->cache);
    s->cache = NULL; s->oldest = NULL;
    s->cache_buckets = 0;
    s->cache_count = s->cache_used = 0;
    s->cache_budget = budget;
    return 0;
}
int meft_rows_set_cache_policy(meft_rows *s, int rows) {
    if (!s || (rows != 0 && rows != 1)) return -1;
    meft_rows_set_cache_bytes(s, s->cache_budget);
    s->cache_rows = rows;
    return 0;
}
static size_t cache_bucket(uint64_t block, size_t buckets) {
    /* Mix sparse/nonconsecutive block IDs; bucket counts are powers of two. */
    block ^= block >> 30; block *= UINT64_C(0xbf58476d1ce4e5b9);
    block ^= block >> 27; block *= UINT64_C(0x94d049bb133111eb);
    block ^= block >> 31;
    return (size_t)block & (buckets - 1);
}
static void cache_unlink(meft_rows *s, struct cached_block *entry) {
    if (entry->previous) entry->previous->next = entry->next;
    else s->newest = entry->next;
    if (entry->next) entry->next->previous = entry->previous;
    else s->oldest = entry->previous;
}
static void cache_promote(meft_rows *s, struct cached_block *entry) {
    entry->previous = NULL; entry->next = s->newest;
    if (s->newest) s->newest->previous = entry;
    else s->oldest = entry;
    s->newest = entry;
}
static void cache_resize(meft_rows *s) {
    if (s->cache_buckets && s->cache_count < s->cache_buckets) return;
    if (s->cache_buckets > SIZE_MAX / 2) return;
    size_t count = s->cache_buckets ? s->cache_buckets * 2 : 16;
    if (count > SIZE_MAX / sizeof(*s->cache)) return;
    struct cached_block **buckets = calloc(count, sizeof(*buckets));
    if (!buckets) return; /* Keep the existing index on optional allocation failure. */
    for (struct cached_block *entry = s->newest; entry; entry = entry->next) {
        size_t bucket = cache_bucket(entry->key, count);
        entry->hash_next = buckets[bucket]; buckets[bucket] = entry;
    }
    free(s->cache); s->cache = buckets; s->cache_buckets = count;
}
static struct cached_block *cache_lookup(meft_rows *s, uint64_t key) {
    if (!s->cache_buckets) return NULL;
    for (struct cached_block *entry = s->cache[cache_bucket(key, s->cache_buckets)];
         entry; entry = entry->hash_next) {
        if (entry->key == key) return entry;
    }
    return NULL;
}
static const unsigned char *cache_find(meft_rows *s, uint64_t key) {
    struct cached_block *entry = cache_lookup(s, key);
    if (!entry) return NULL;
    if (entry != s->newest) { cache_unlink(s, entry); cache_promote(s, entry); }
    if (entry->query != s->cache_query) {
        entry->query = s->cache_query;
        if (s->cache_rows) ++s->last_row_hits;
        else ++s->last_hits;
    }
    return entry->bytes;
}
static void cache_insert(meft_rows *s, uint64_t key, size_t size, const unsigned char *bytes) {
    if (size > s->cache_budget || size > SIZE_MAX - sizeof(struct cached_block)) return;
    /* Repeated rows in a freshly decoded group must not create duplicate keys
     * or count as hits on data that was absent at query start. */
    struct cached_block *existing = s->cache_rows ? cache_lookup(s, key) : NULL;
    if (existing) {
        if (existing != s->newest) { cache_unlink(s, existing); cache_promote(s, existing); }
        return;
    }
    while (s->oldest && s->cache_used > s->cache_budget - size) {
        struct cached_block *old = s->oldest;
        struct cached_block **link = s->cache + cache_bucket(old->key, s->cache_buckets);
        while (*link != old) link = &(*link)->hash_next;
        *link = old->hash_next;
        cache_unlink(s, old); --s->cache_count;
        s->cache_used -= old->size;
        free(old);
    }
    /* Evict before allocation to respect the payload budget even transiently.
     * Cache allocation is optional; failure leaves a valid scratch result. */
    cache_resize(s);
    if (!s->cache_buckets) return;
    struct cached_block *copy = malloc(sizeof(*copy) + size);
    if (!copy) return;
    copy->key = key; copy->size = size; copy->query = 0;
    memcpy(copy->bytes, bytes, size);
    size_t bucket = cache_bucket(key, s->cache_buckets);
    copy->hash_next = s->cache[bucket]; s->cache[bucket] = copy;
    cache_promote(s, copy);
    ++s->cache_count;
    s->cache_used += size;
}
static int order(const void *a, const void *b) {
    const struct request *x = a, *y = b;
    return x->block < y->block ? -1 : x->block > y->block;
}
int meft_rows_gather(meft_rows *s, const uint64_t *ids, size_t count,
                     void *out, size_t outbytes, char *err, size_t cap) {
    if (!s) return fail(err, cap, "null store");
    s->last_blocks = s->last_decodes = s->last_hits = s->last_row_hits = 0;
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
    if (s->cache_count) {
        if (!++s->cache_query) {
            for (struct cached_block *entry = s->newest; entry; entry = entry->next) entry->query = 0;
            s->cache_query = 1;
        }
        /* Validate every ID above before touching output/cache. Consume all
         * existing hits before misses can evict a block needed later in this
         * call. Fully cached queries skip sorting and decompression entirely. */
        size_t pending = 0;
        for (size_t i = 0; i < count; ++i) {
            uint64_t key = s->cache_rows ? requests[i].block * s->block_rows + requests[i].within : requests[i].block;
            const unsigned char *bytes = cache_find(s, key);
            if (bytes) {
                memcpy((unsigned char *)out + requests[i].position * s->width,
                       bytes + (s->cache_rows ? 0 : requests[i].within * s->width), (size_t)s->width);
            } else requests[pending++] = requests[i];
        }
        count = pending;
    }
    if (count > 1) qsort(requests, count, sizeof(*requests), order);
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
            if (!s->cache_rows) cache_insert(s, block, (size_t)raw, bytes);
        }
        do {
            const unsigned char *row = bytes + requests[i].within * s->width;
            memcpy((unsigned char *)out + requests[i].position * s->width,
                   row, (size_t)s->width);
            if (codec && s->cache_rows)
                cache_insert(s, block * s->block_rows + requests[i].within, (size_t)s->width, row);
            ++i;
        } while (i < count && requests[i].block == block);
    }
done:
    free(requests);
    return reason ? fail(err, cap, reason) : 0;
}

int meft_rows_gather_chunked(meft_rows *s, const uint64_t *ids, size_t count,
                             void *out, size_t outbytes, size_t max_query_rows,
                             char *err, size_t cap) {
    if (!s) return fail(err, cap, "null store");
    s->last_blocks = s->last_decodes = s->last_hits = s->last_row_hits = 0;
    if (!max_query_rows || count > SIZE_MAX / s->width ||
        count * s->width > outbytes || count > SIZE_MAX / sizeof(uint64_t) ||
        (count && (!ids || !out)))
        return fail(err, cap, "invalid output capacity, query size or chunk limit");
    uint64_t blocks = 0, decodes = 0, hits = 0, row_hits = 0;
    int status = 0;
    for (size_t start = 0; start < count;) {
        size_t n = count - start < max_query_rows ? count - start : max_query_rows;
        status = meft_rows_gather(s, ids + start, n,
                                  (unsigned char *)out + start * (size_t)s->width,
                                  n * (size_t)s->width, err, cap);
        blocks += s->last_blocks; decodes += s->last_decodes; hits += s->last_hits; row_hits += s->last_row_hits;
        if (status) break;
        start += n;
    }
    s->last_blocks = blocks; s->last_decodes = decodes; s->last_hits = hits; s->last_row_hits = row_hits;
    return status;
}

static float row_float(const unsigned char *p, int dtype) {
    uint32_t bits;
    if (dtype == 1) {
        bits = (uint32_t)p[0] | (uint32_t)p[1] << 8 |
               (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    } else {
        uint32_t half = (uint32_t)p[0] | (uint32_t)p[1] << 8;
        if (dtype == 3) {
            bits = half << 16;
        } else {
            uint32_t sign = (half & 0x8000u) << 16;
            uint32_t exponent = (half >> 10) & 31u, mantissa = half & 1023u;
            if (exponent == 31) bits = sign | 0x7f800000u | mantissa << 13;
            else if (exponent) bits = sign | (exponent + 112u) << 23 | mantissa << 13;
            else if (!mantissa) bits = sign;
            else {
                unsigned shift = 0;
                while (!(mantissa & 1024u)) { mantissa <<= 1; ++shift; }
                bits = sign | (113u - shift) << 23 | (mantissa & 1023u) << 13;
            }
        }
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int embedding_bag(meft_rows *s, const uint64_t *ids, size_t count,
                           const uint64_t *offsets, size_t bags, int dtype, int mean,
                           uint64_t padding_id, size_t chunk_rows,
                           float *out, size_t out_floats, char *err, size_t cap,
                           int weighted, const unsigned char *weights, size_t weights_bytes) {
    if (!s) return fail(err, cap, "null store");
    s->last_blocks = s->last_decodes = s->last_hits = s->last_row_hits = 0;
    size_t element = dtype == 1 ? 4 : 2;
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 ||
        dtype < 1 || dtype > 3 || (mean != 0 && mean != 1) || !chunk_rows ||
        s->width > SIZE_MAX || s->width % element)
        return fail(err, cap, "invalid bag dtype, mode, chunk size or row width");
    if (weighted && (mean || count > SIZE_MAX / element ||
                     count * element > weights_bytes || (count && !weights)))
        return fail(err, cap, "invalid per-sample weights capacity or mode");
    size_t dim = (size_t)(s->width / element);
    if (bags > SIZE_MAX / dim || bags * dim > out_floats ||
        dim > SIZE_MAX / sizeof(double) || bags > SIZE_MAX / (dim * sizeof(float)) ||
        bags > SIZE_MAX / sizeof(uint64_t) - 1 || count > SIZE_MAX / sizeof(uint64_t) ||
        (bags && (!offsets || !out)) ||
        (count && (!ids || !bags)) ||
        (padding_id != UINT64_MAX && padding_id >= s->nrows))
        return fail(err, cap, "invalid bag capacity, query or padding ID");
    if (!bags) return 0;
    if (offsets[0] != 0 || offsets[bags] != count)
        return fail(err, cap, "bag offsets must span all IDs");
    for (size_t b = 0; b < bags; ++b)
        if (offsets[b] > offsets[b + 1] || offsets[b + 1] > count)
            return fail(err, cap, "bag offsets must be nondecreasing and bounded");
    for (size_t i = 0; i < count; ++i)
        if (ids[i] >= s->nrows) return fail(err, cap, "row ID out of bounds");
    size_t staged = count < chunk_rows ? count : chunk_rows;
    if (staged > SIZE_MAX / s->width)
        return fail(err, cap, "bag staging capacity overflow");
    unsigned char *rows = staged ? malloc(staged * (size_t)s->width) : NULL;
    double *sum = calloc(dim, sizeof(*sum));
    if ((staged && !rows) || !sum) {
        free(rows); free(sum);
        return fail(err, cap, "allocation failed");
    }
    uint64_t blocks = 0, decodes = 0, hits = 0, row_hits = 0;
    int status = 0;
    size_t bag = 0, observed = 0;
    /* Coalesce blocks across bag boundaries, then consume restored ID order.
     * Only the active bag's FP64 accumulator and chunk-sized rows are retained.
     * Chunk boundaries never round partial sums or reset a partially read bag. */
    for (size_t start = 0; start < count;) {
        size_t n = count - start < staged ? count - start : staged;
        status = meft_rows_gather(s, ids + start, n, rows, n * (size_t)s->width, err, cap);
        blocks += s->last_blocks; decodes += s->last_decodes; hits += s->last_hits; row_hits += s->last_row_hits;
        if (status) goto done_bags;
        for (size_t i = 0; i < n; ++i) {
            while (bag < bags && offsets[bag + 1] <= start + i) {
                double divisor = mean && observed ? (double)observed : 1.0;
                for (size_t j = 0; j < dim; ++j) out[bag * dim + j] = (float)(sum[j] / divisor);
                ++bag; observed = 0;
                memset(sum, 0, dim * sizeof(*sum));
            }
            if (ids[start + i] == padding_id) continue;
            ++observed;
            const unsigned char *row = rows + i * (size_t)s->width;
            double scale = weighted ? row_float(weights + (start + i) * element, dtype) : 1.0;
            for (size_t j = 0; j < dim; ++j)
                sum[j] += (double)row_float(row + j * element, dtype) * scale;
        }
        start += n;
    }
    while (bag < bags) {
        double divisor = mean && observed ? (double)observed : 1.0;
        for (size_t j = 0; j < dim; ++j) out[bag * dim + j] = (float)(sum[j] / divisor);
        ++bag; observed = 0;
        memset(sum, 0, dim * sizeof(*sum));
    }
done_bags:
    s->last_blocks = blocks; s->last_decodes = decodes; s->last_hits = hits; s->last_row_hits = row_hits;
    free(rows); free(sum);
    return status;
}

int meft_rows_embedding_bag(meft_rows *s, const uint64_t *ids, size_t count,
                           const uint64_t *offsets, size_t bags, int dtype, int mean,
                           uint64_t padding_id, size_t chunk_rows,
                           float *out, size_t out_floats, char *err, size_t cap) {
    return embedding_bag(s, ids, count, offsets, bags, dtype, mean, padding_id,
                         chunk_rows, out, out_floats, err, cap, 0, NULL, 0);
}

int meft_rows_embedding_bag_weighted(meft_rows *s, const uint64_t *ids, size_t count,
                                    const uint64_t *offsets, size_t bags, int dtype,
                                    const void *weights, size_t weights_bytes,
                                    uint64_t padding_id, size_t chunk_rows,
                                    float *out, size_t out_floats, char *err, size_t cap) {
    return embedding_bag(s, ids, count, offsets, bags, dtype, 0, padding_id,
                         chunk_rows, out, out_floats, err, cap, 1, weights, weights_bytes);
}
