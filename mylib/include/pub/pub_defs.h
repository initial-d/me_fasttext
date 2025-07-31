#ifndef PUB_DEFS_H_
#define PUB_DEFS_H_
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

#define MAX_LINE_LEN			(10240)
#define MAX_URI_LEN				(256)
#define MAX_CONFIG_LEN			(512)
#define MAX_WORD_LEN			(1024)
#define MAX_SPELL_LEN			(3076)

#define min2(a, b)				((a) < (b) ? (a) : (b))
#define min3(a, b, c)			((c) < ((a) < (b) ? (a) : (b)) ? (c) : ((a) < (b) ? (a) : (b)))
#define max2(a, b)				((a) > (b) ? (a) : (b))
#define max3(a, b, c)			((c) > ((a) > (b) ? (a) : (b)) ? (c) : ((a) > (b) ? (a) : (b)))
#define swap2(a, b)				{a ^= b; b ^= a; a ^= b;}
#define abs1(x)					((x) > 0 ? (x) : -(x))

#ifdef _WIN32
#define FILE_SEP				('\\')
#define LIKELY(x)				(x)
#define UNLIKELY(x)				(x)
#else
#define FILE_SEP				('/')
#define LIKELY(x)				__builtin_expect(!!(x), 1)
#define UNLIKELY(x)				__builtin_expect(!!(x), 0)
#endif

typedef char				Bool;
typedef char				int8;
typedef unsigned char		uint8;
typedef short				int16;
typedef unsigned short		uint16;
typedef int					int32;
typedef unsigned int		uint32;
typedef long long			int64;
typedef unsigned long long	uint64;
typedef struct
{
	uint64 low64;
	uint64 high64;
} uint128;

#define isoverflow_int8(x)		(((int64)(x) < -0x80) || ((int64)(x) > 0x7f))
#define isoverflow_uint8(x)		(((int64)(x) < 0) || ((int64)(x) > 0xff))
#define isoverflow_int16(x)		(((int64)(x) < -0x8000) || ((int64)(x) > 0x7fff))
#define isoverflow_uint16(x)	(((int64)(x) < 0) || ((int64)(x) > 0xffff))
#define isoverflow_int32(x)		(((int64)(x) < -0x80000000LL) || ((int64)(x) > 0x7fffffffLL))
#define isoverflow_uint32(x)	(((int64)(x) < 0) || ((int64)(x) > 0xffffffffLL))

#ifdef __cplusplus
}
#endif

#endif
