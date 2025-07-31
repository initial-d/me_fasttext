#ifndef PUB_COMBUF_H_
#define PUB_COMBUF_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COMBUF_FLAG_FIXED			(2)

#define combuf_delete(combuf)		{combuf_delete_unsafe(combuf); combuf = NULL;}

typedef struct _ComBuf ComBuf;
typedef int (*ComBufWriter)(char* dest, const void* data);
typedef void* (*ComBufReader)(const char* src, void* o_data);

ComBuf*	combuf_new(int flag, ComBufWriter writer, ComBufReader reader);
ComBuf*	combuf_clone(ComBuf* combuf);
void	combuf_clear(ComBuf* combuf);
void	combuf_delete_unsafe(ComBuf* combuf);
void	combuf_set_writer(ComBuf* combuf, ComBufWriter writer);
void	combuf_set_reader(ComBuf* combuf, ComBufReader reader);
long	combuf_set(ComBuf* combuf, const void* data, size_t len);
long	combuf_mod(ComBuf* combuf, size_t offset, const void* data, size_t len);
void*	combuf_get(ComBuf* combuf, size_t offset, void* o_data);
long	combuf_store_to_stream(ComBuf* combuf, FILE* stream);
ComBuf*	combuf_load_from_stream(FILE* stream, size_t* o_pos);
void*	combuf_map_to_buffer(ComBuf* combuf, size_t* o_size);
ComBuf*	combuf_map_from_buffer(void* buffer, size_t* o_pos);

#ifdef __cplusplus
}
#endif

#endif
