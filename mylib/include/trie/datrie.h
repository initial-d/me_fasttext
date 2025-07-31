#ifndef DATRIE_H_
#define DATRIE_H_

#include <stdio.h>
#include "trie/trie_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define datrie_delete(datrie)			{datrie_delete_unsafe(datrie);(datrie) = NULL;}

typedef struct _DATrie DATrie;

DATrie*	datrie_new(void);
DATrie*	datrie_clone(DATrie* datrie);
void	datrie_clear(DATrie* datrie);
void	datrie_delete_unsafe(DATrie* datrie);
long	datrie_store_to_stream(DATrie* datrie, FILE* stream);
DATrie*	datrie_load_from_stream(FILE* stream, size_t* o_pos);
long	datrie_store_to_file(DATrie* datrie, const char* uri);
DATrie*	datrie_load_from_file(const char* uri);
void*	datrie_map_to_buffer(DATrie* datrie, size_t* o_size);
DATrie*	datrie_map_from_buffer(void* buffer, size_t* o_pos);
int		datrie_insert_key(DATrie* datrie, const char* key, size_t value);
int		datrie_insert_file(DATrie* datrie, const char* uri);
int		datrie_remove_key(DATrie* datrie, const char* key);
int		datrie_remove_file(DATrie* datrie, const char* uri);
void	datrie_print_keys(DATrie* datrie, FILE* stream);
void	datrie_show_keys(DATrie* datrie, const char* uri);
int		datrie_foreach_keys(DATrie* datrie, TrieForeach foreach, void* toforeach);
void	datrie_show_info(DATrie* datrie, FILE* stream);
size_t	datrie_get_key_num(DATrie* datrie);
int		datrie_search_exact_match(DATrie* datrie, const char* key, size_t* o_value);
int		datrie_search_prefix_match(DATrie* datrie, const char* key, TrieCompare compare, void* tocompare,
		size_t* o_values, size_t size);
int		datrie_search_approx_match(DATrie* datrie, const char* key, TrieCompare compare, void* tocompare,
		size_t* o_values, size_t size);
int		datrie_search_prefix_split(DATrie* datrie, const char* key, size_t* o_value);
int		datrie_search_common_split(DATrie* datrie, const char* key, size_t* o_values, size_t size);
int		datrie_search_forwardmm_compose(DATrie* datrie, const char* key, size_t* o_values, size_t size);
int		datrie_search_minpath_compose(DATrie* datrie, const char* key, size_t* o_values, size_t size);
#ifndef _WIN32
DATrie*	datrie_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int		datrie_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
