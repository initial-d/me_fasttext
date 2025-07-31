#ifndef MD5_H_
#define MD5_H_

#include "pub/pub.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32	md5hash(const char* str, uint32 len);
uint64	md5hash64(const char* str, uint32 len);
uint128	md5hash128(const char* str, uint32 len);
void	md5_string(const char* str, char digest[33]);
void	md5_file(const char* uri, char digest[33]);

#ifdef __cplusplus
}
#endif

#endif
