#ifndef HASHTABLE_H_
#define HASHTABLE_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define hashtable_delete(hashtable)			{hashtable_delete_unsafe(hashtable); hashtable = NULL;}

typedef struct _HashTable HashTable;

HashTable*	hashtable_new(size_t rows, size_t cols, size_t payload);
HashTable*	hashtable_clone(HashTable* hashtable);
void		hashtable_clear(HashTable* hashtable);
void		hashtable_delete_unsafe(HashTable* hashtable);
long		hashtable_store_to_stream(HashTable* hashtable, FILE* stream);
HashTable*	hashtable_load_from_stream(FILE* stream, size_t* o_pos);
long		hashtable_store_to_file(HashTable* hashtable, const char* uri);
HashTable*	hashtable_load_from_file(const char* uri);
void*		hashtable_map_to_buffer(HashTable* hashtable, size_t* o_size);
HashTable*	hashtable_map_from_buffer(void* buffer, size_t* o_pos);
int			hashtable_insert_key(HashTable* hashtable, const char* key, size_t value);
int			hashtable_insert_file(HashTable* hashtable, const char* uri);
int			hashtable_remove_key(HashTable* hashtable, const char* key);
int			hashtable_remove_file(HashTable* hashtable, const char* uri);
int			hashtable_get_value(HashTable* hashtable, const char* key, size_t* o_value);
size_t		hashtable_get_key_num(HashTable* hashtable);
#ifndef _WIN32
HashTable*	hashtable_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int			hashtable_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
