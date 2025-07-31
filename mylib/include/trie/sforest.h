#ifndef SFOREST_H_
#define SFOREST_H_

#include <stdio.h>
#include "trie/trie_defs.h"
#include "trie/dastrie.h"
#include "trie/forest.h"

#ifdef __cplusplus
extern "C" {
#endif

#define sforest_delete(sforest)				{sforest_delete_unsafe(sforest);(sforest) = NULL;}

typedef struct _SForest SForest;
struct _SForest
{
	DASTrie**		nodes;
	size_t			size;
	unsigned int	flag;
};

SForest*	sforest_new(size_t size);
SForest*	sforest_clone(SForest* sforest);
void		sforest_clear(SForest* sforest);
void		sforest_delete_unsafe(SForest* sforest);
long		sforest_store_to_stream(SForest* sforest, FILE* stream);
SForest*	sforest_load_from_stream(FILE* stream, size_t* o_pos);
long		sforest_store_to_file(SForest* sforest, const char* uri);
SForest*	sforest_load_from_file(const char* uri);
void*		sforest_map_to_buffer(SForest* sforest, size_t* o_size);
SForest*	sforest_map_from_buffer(void* buffer, size_t* o_pos);
int			sforest_build_file(SForest* sforest, const char* uri);
int			sforest_build_convert(SForest* sforest, Forest* forest);
void		sforest_show_keys(SForest* sforest, const char* uri);
void		sforest_print_keys(SForest* sforest, FILE* file);
int			sforest_foreach_keys(SForest* sforest, TrieForeach forest, void* toforeach);
void		sforest_show_info(SForest* sforest, FILE* stream);
size_t		sforest_get_key_num(SForest* sforest);
int			sforest_search_exact_match(SForest* sforest, const char* key, size_t* o_value);
int			sforest_search_approx_match(SForest* sforest, const char* key, TrieCompare compare, void* tocompare,
			size_t* o_values, size_t size);
int			sforest_search_prefix_split(SForest* sforest, const char* key, size_t* o_value);
int			sforest_search_common_split(SForest* sforest, const char* key, size_t* o_values, size_t size);
#ifndef _WIN32
SForest*	sforest_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int			sforest_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
