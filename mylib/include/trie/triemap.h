#ifndef TRIEMAP_H_
#define TRIEMAP_H_

#include <stdio.h>
#include "trie/trie_defs.h"
#include "trie/datrie.h"

#ifdef __cplusplus
extern "C" {
#endif

#define	TRIEMAP_FLAG_FIXED				COMBUF_FLAG_FIXED
#define triemap_delete(triemap)			{triemap_delete_unsafe(triemap);(triemap) = NULL;}

typedef struct _TrieMap TrieMap;
struct _TrieMap
{
	DATrie*			datrie;
	ComBuf*			combuf;
	unsigned int	flag;
};

TrieMap*	triemap_new(int flag, TrieMapWriter writer, TrieMapReader reader);
TrieMap*	triemap_clone(TrieMap* triemap);
void		triemap_clear(TrieMap* triemap);
void		triemap_delete_unsafe(TrieMap* triemap);
long		triemap_store_to_stream(TrieMap* triemap, FILE* stream);
TrieMap*	triemap_load_from_stream(FILE* stream, size_t* o_pos);
long		triemap_store_to_file(TrieMap* triemap, const char* uri);
TrieMap*	triemap_load_from_file(const char* uri);
void*		triemap_map_to_buffer(TrieMap* triemap, size_t* o_size);
TrieMap*	triemap_map_from_buffer(void* buffer, size_t* o_pos);
void		triemap_set_writer(TrieMap* triemap, TrieMapWriter writer);
void		triemap_set_reader(TrieMap* triemap, TrieMapReader reader);
int			triemap_insert_file(TrieMap* triemap, const char* uri);
int			triemap_remove_file(TrieMap* triemap, const char* uri);
void		triemap_export_file(TrieMap* triemap, const char* uri, TrieMapPrint print);
int			triemap_set(TrieMap* triemap, const char* key, const void* value, size_t len);
int			triemap_unset(TrieMap* triemap, const char* key);
void*		triemap_get(TrieMap* triemap, const char* key, void* o_data);
size_t		triemap_get_key_num(TrieMap* triemap);
int			triemap_parse_sxml(TrieMap* triemap, char* sxml);
int			triemap_parse_request(TrieMap* triemap, char* request);
#ifndef _WIN32
TrieMap*	triemap_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int			triemap_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
