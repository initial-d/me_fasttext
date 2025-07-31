#ifndef STRIEMAP_H_
#define STRIEMAP_H_

#include <stdio.h>
#include "trie/trie_defs.h"
#include "trie/triemap.h"
#include "trie/dastrie.h"

#ifdef __cplusplus
extern "C" {
#endif

#define	STRIEMAP_FLAG_FIXED				COMBUF_FLAG_FIXED
#define striemap_delete(striemap)		{striemap_delete_unsafe(striemap);(striemap) = NULL;}

typedef struct _STrieMap STrieMap;
struct _STrieMap
{
	DASTrie*		dastrie;
	ComBuf*			combuf;
	unsigned int	flag;
};

STrieMap*	striemap_new(int flag, TrieMapWriter writer, TrieMapReader reader);
STrieMap*	striemap_clone(STrieMap* striemap);
void		striemap_clear(STrieMap* striemap);
void		striemap_delete_unsafe(STrieMap* striemap);
long		striemap_store_to_stream(STrieMap* striemap, FILE* stream);
STrieMap*	striemap_load_from_stream(FILE* stream, size_t* o_pos);
long		striemap_store_to_file(STrieMap* striemap, const char* uri);
STrieMap*	striemap_load_from_file(const char* uri);
void*		striemap_map_to_buffer(STrieMap* striemap, size_t* o_size);
STrieMap*	striemap_map_from_buffer(void* buffer, size_t* o_pos);
void		striemap_set_writer(STrieMap* striemap, TrieMapWriter writer);
void		striemap_set_reader(STrieMap* striemap, TrieMapReader reader);
int			striemap_build_keyset(STrieMap* striemap, const char** keys, const void** values,
			size_t* lens, size_t size);
int			striemap_build_file(STrieMap* striemap, const char* uri);
int			striemap_build_convert(STrieMap* striemap, TrieMap* triemap);
void		striemap_export_file(STrieMap* striemap, const char* uri, TrieMapPrint print);
void*		striemap_get(STrieMap* striemap, const char* key, void* o_data);
size_t		striemap_get_key_num(STrieMap* striemap);
#ifndef _WIN32
STrieMap*	striemap_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int			striemap_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
