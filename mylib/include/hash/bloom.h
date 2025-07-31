#ifndef BLOOM_H_
#define BLOOM_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define bloom_delete(bloom)			{bloom_delete_unsafe(bloom);bloom = NULL;}

typedef struct _Bloom Bloom;

Bloom*	bloom_new(size_t size, size_t kfunc);
Bloom*	bloom_clone(Bloom* bloom);
void	bloom_clear(Bloom* bloom);
void	bloom_delete_unsafe(Bloom* bloom);
long	bloom_store_to_stream(Bloom* bloom, FILE* stream);
Bloom*	bloom_load_from_stream(FILE* stream, size_t* o_pos);
long	bloom_store_to_file(Bloom* bloom, const char* uri);
Bloom*	bloom_load_from_file(const char* uri);
void*	bloom_map_to_buffer(Bloom* bloom, size_t* o_size);
Bloom*	bloom_map_from_buffer(void* buffer, size_t* o_pos);
int		bloom_insert_file(Bloom* bloom, const char* uri);
int		bloom_insert(Bloom* bloom, const char* key);
int		bloom_check(Bloom* bloom, const char* key);
size_t	bloom_get_key_num(Bloom* bloom);
size_t	bloom_calculate_mke(size_t n, double expect_e, size_t* o_m, int* o_k, double* o_actual_e);
#ifndef _WIN32
Bloom*	bloom_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int		bloom_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
