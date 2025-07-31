#ifndef FOREST_H_
#define FOREST_H_

#include <stdio.h>
#include "trie/trie_defs.h"
#include "trie/datrie.h"

#ifdef __cplusplus
extern "C" {
#endif

#define forest_delete(forest)				{forest_delete_unsafe(forest);(forest) = NULL;}

typedef struct _Forest Forest;
struct _Forest
{
	DATrie**		nodes;
	size_t			size;
	unsigned int	flag;
};

Forest*	forest_new(size_t size);
Forest*	forest_clone(Forest* forest);
void	forest_clear(Forest* forest);
void	forest_delete_unsafe(Forest* forest);
long	forest_store_to_stream(Forest* forest, FILE* stream);
Forest*	forest_load_from_stream(FILE* stream, size_t* o_pos);
long	forest_store_to_file(Forest* forest, const char* uri);
Forest*	forest_load_from_file(const char* uri);
void*	forest_map_to_buffer(Forest* forest, size_t* o_size);
Forest*	forest_map_from_buffer(void* buffer, size_t* o_pos);
int		forest_insert_key(Forest* forest, const char* key, size_t value);
int		forest_insert_file(Forest* forest, const char* uri);
int		forest_remove_key(Forest* forest, const char* key);
int		forest_remove_file(Forest* forest, const char* uri);
void	forest_show_keys(Forest* forest, const char* uri);
void	forest_print_keys(Forest* forest, FILE* file);
int		forest_foreach_keys(Forest* forest, TrieForeach foreach, void* toforeach);
void	forest_show_info(Forest* forest, FILE* stream);
size_t	forest_get_key_num(Forest* forest);
int		forest_search_exact_match(Forest* forest, const char* key, size_t* o_value);
int		forest_search_approx_match(Forest* forest, const char* key, TrieCompare compare, void* tocompare,
		size_t* o_values, size_t size);
int		forest_search_prefix_split(Forest* forest, const char* key, size_t* o_value);
int		forest_search_common_split(Forest* forest, const char* key, size_t* o_values, size_t size);
#ifndef _WIN32
Forest*	forest_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int		forest_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
