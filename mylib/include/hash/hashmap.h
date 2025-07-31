#ifndef HASHMAP_H_
#define HASHMAP_H_

#include <stdio.h>
#include "pub/pub_combuf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HASHMAP_FLAG_FIXED				COMBUF_FLAG_FIXED

#define hashmap_delete(hashmap)			{hashmap_delete_unsafe(hashmap); hashmap = NULL;}

typedef struct _HashMap HashMap;
typedef ComBufWriter HashMapWriter;
typedef ComBufReader HashMapReader;

HashMap*	hashmap_new(size_t rows, size_t cols, size_t payload, int flag,
			HashMapWriter writer, HashMapReader reader);
HashMap*	hashmap_clone(HashMap* hashmap);
void		hashmap_clear(HashMap* hashmap);
void		hashmap_delete_unsafe(HashMap* hashmap);
long		hashmap_store_to_stream(HashMap* hashmap, FILE* stream);
HashMap*	hashmap_load_from_stream(FILE* stream, size_t* o_pos);
long		hashmap_store_to_file(HashMap* hashmap, const char* uri);
HashMap*	hashmap_load_from_file(const char* uri);
void*		hashmap_map_to_buffer(HashMap* hashmap, size_t* o_size);
HashMap*	hashmap_map_from_buffer(void* buffer, size_t* o_pos);
void		hashmap_set_writer(HashMap* hashmap, HashMapWriter writer);
void		hashmap_set_reader(HashMap* hashmap, HashMapReader reader);
int			hashmap_insert_file(HashMap* hashmap, const char* uri);
int			hashmap_remove_file(HashMap* hashmap, const char* uri);
int			hashmap_set(HashMap* hashmap, const char* key, const void* value, size_t len);
int			hashmap_unset(HashMap* hashmap, const char* key);
void*		hashmap_get(HashMap* hashmap, const char* key, void* o_value);
size_t		hashmap_get_key_num(HashMap* hashmap);
#ifndef _WIN32
HashMap*	hashmap_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int			hashmap_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
