#ifndef PUB_MALLOC_H_
#define PUB_MALLOC_H_

#include <stdio.h>
#include <malloc.h>

#ifndef _WIN32
#include <alloca.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define pub_malloc(size)				pub_malloc_detail((size), __FILE__, __FUNCTION__, __LINE__)
#define pub_calloc(num, size)			pub_calloc_detail((num), (size), __FILE__, __FUNCTION__, __LINE__)
#define pub_realloc(ptr, size)			pub_realloc_detail((ptr), (size), __FILE__, __FUNCTION__, __LINE__)
#define pub_alloca(size)				alloca(size)
#define pub_free(ptr)					{free((void*)(ptr));(ptr) = NULL;}

void*	pub_malloc_detail(size_t size, char* file_name, const char* function_name, int line_num);
void*	pub_calloc_detail(size_t num, size_t size, char* file_name, const char* function_name, int line_num);
void*	pub_realloc_detail(void* ptr, size_t size, char* file_name, const char* function_name, int line_num);

#ifdef __cplusplus
}
#endif

#endif
