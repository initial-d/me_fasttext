#ifndef MURMURHASH_H_
#define MURMURHASH_H_

#include "pub/pub.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32	murmurhash(const char* str, uint32 len, uint32 seed);
uint64	murmurhash64(const char* str, uint32 len, uint32 seed);
uint128	murmurhash128(const char* str, uint32 len, uint32 seed);

#ifdef __cplusplus
}
#endif

#endif
