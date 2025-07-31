#ifndef GENERALHASH_H_
#define GENERALHASH_H_

#include "pub/pub.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32	rshash(const char* str, uint32 len);
uint32	jshash(const char* str, uint32 len);
uint32	pjwhash(const char* str, uint32 len);
uint32	elfhash(const char* str, uint32 len);
uint32	bkdrhash(const char* str, uint32 len);
uint32	sdbmhash(const char* str, uint32 len);
uint32	djbhash(const char* str, uint32 len);
uint32	dekhash(const char* str, uint32 len);
uint32	fnvhash(const char* str, uint32 len);
uint32	aphash(const char* str, uint32 len);

#ifdef __cplusplus
}
#endif

#endif
