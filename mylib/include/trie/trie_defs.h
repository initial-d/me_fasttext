#ifndef TRIE_DEFS_H_
#define TRIE_DEFS_H_

#include <stdio.h>
#include "pub/pub_combuf.h"

#ifdef __cplusplus
extern "C" {
#endif
	
#define TRIE_ALPHASIZE							(256)
#define TRIE_POOL_BEGIN							(1)
#define trie_alphatolabel(alpha)				((unsigned char)(alpha))
#define trie_labeltoalpha(label)				((char)(label))
#define trie_hash(key, size)					(*(uint16*)(key) % (size))

typedef ComBufWriter TrieMapWriter;
typedef ComBufReader TrieMapReader;
typedef void (*TrieMapPrint)(FILE* stream, const char* key, void* buf);
typedef int (*TrieForeach)(const char* key, size_t value, void* udata);
typedef int (*TrieCompare)(size_t value1, size_t value2, void* udata);
typedef struct _Trie_Labels Trie_Labels;
struct _Trie_Labels
{
	unsigned short	num;
	unsigned char	labels[TRIE_ALPHASIZE];
};

#ifdef __cplusplus
}
#endif

#endif
