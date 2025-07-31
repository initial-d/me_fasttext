#ifndef DASTRIE_H_
#define DASTRIE_H_

#include <stdio.h>
#include "trie/trie_defs.h"
#include "trie/datrie.h"

#ifdef __cplusplus
extern "C" {
#endif

#define dastrie_delete(dastrie)				{dastrie_delete_unsafe(dastrie);dastrie = NULL;}

typedef struct _DASTrie DASTrie;

DASTrie*	dastrie_new(void);
DASTrie*	dastrie_clone(DASTrie* dastrie);
void		dastrie_clear(DASTrie* dastrie);
void		dastrie_delete_unsafe(DASTrie* dastrie);
long		dastrie_store_to_stream(DASTrie* dastrie, FILE* stream);
DASTrie*	dastrie_load_from_stream(FILE* stream, size_t* o_pos);
long		dastrie_store_to_file(DASTrie* dastrie, const char* uri);
DASTrie*	dastrie_load_from_file(const char* uri);
void*		dastrie_map_to_buffer(DASTrie* dastrie, size_t* o_size);
DASTrie*	dastrie_map_from_buffer(void* buffer, size_t* o_pos);
int			dastrie_build_keyset(DASTrie* dastrie, const char** keys, const size_t* values, size_t size);
int			dastrie_build_file(DASTrie* dastrie, const char* uri);
int			dastrie_build_convert(DASTrie* dastrie, DATrie* datrie);
void		dastrie_print_keys(DASTrie* dastrie, FILE* stream);
void		dastrie_show_keys(DASTrie* dastrie, const char* uri);
int			dastrie_foreach_keys(DASTrie* dastrie, TrieForeach foreach, void* toforeach);
void		dastrie_show_info(DASTrie* dastrie, FILE* stream);
size_t		dastrie_get_key_num(DASTrie* dastrie);
int			dastrie_search_exact_match(DASTrie* dastrie, const char* key, size_t* o_value);
int			dastrie_search_prefix_match(DASTrie* dastrie, const char* key, TrieCompare compare, void* tocompare,
			size_t* o_values, size_t size);
int			dastrie_search_approx_match(DASTrie* dastrie, const char* key, TrieCompare compare, void* tocompare,
			size_t* o_values, size_t size);
int			dastrie_search_prefix_split(DASTrie* dastrie, const char* key, size_t* o_value);
int			dastrie_search_common_split(DASTrie* dastrie, const char* key, size_t* o_values, size_t size);
int			dastrie_search_forwardmm_compose(DASTrie* dastrie, const char* key, size_t* o_values, size_t size);
int			dastrie_search_minpath_compose(DASTrie* dastrie, const char* key, size_t* o_values, size_t size);
#ifndef _WIN32
DASTrie*	dastrie_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int			dastrie_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
