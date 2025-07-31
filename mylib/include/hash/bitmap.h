#ifndef BITMAP_H_
#define BITMAP_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define bitmap_delete(bitmap)			{bitmap_delete_unsafe(bitmap);(bitmap) = NULL;}

typedef struct _BitMap BitMap;

BitMap*	bitmap_new(size_t size);
BitMap*	bitmap_clone(BitMap* bitmap);
void	bitmap_clear(BitMap* bitmap);
void	bitmap_delete_unsafe(BitMap* bitmap);
long	bitmap_store_to_stream(BitMap* bitmap, FILE* stream);
BitMap*	bitmap_load_from_stream(FILE* stream, size_t* o_pos);
long	bitmap_store_to_file(BitMap* bitmap, const char* uri);
BitMap*	bitmap_load_from_file(const char* uri);
void*	bitmap_map_to_buffer(BitMap* bitmap, size_t* o_size);
BitMap*	bitmap_map_from_buffer(void* buffer, size_t* o_pos);
int		bitmap_set(BitMap* bitmap, size_t offset);
int		bitmap_reset(BitMap* bitmap, size_t offset);
int		bitmap_get(BitMap* bitmap, size_t offset);
#ifndef _WIN32
BitMap*	bitmap_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int		bitmap_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
