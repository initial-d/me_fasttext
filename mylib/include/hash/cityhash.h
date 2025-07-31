#ifndef PUB_HASH_H_
#define PUB_HASH_H_

#include "pub/pub.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32	cityhash(const char* str, uint32 len);
uint64	cityhash64(const char* str, uint32 len);
uint128	cityhash128(const char* str, uint32 len);
uint64	cityhash64_with_seed(const char* str, uint32 len, uint32 seed);

#ifdef __cplusplus
}
#endif

#endif
