#ifndef PUB_SPRINTF_H_
#define PUB_SPRINTF_H_

#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define pub_snprintf(dest, size, format, ...)		_snprintf((dest), (size), (format), __VA_ARGS__)
#define pub_vsnprintf(dest, size, format, va_ptr)	_vsnprintf((dest), (size), (format), (va_ptr))

#else
#define pub_snprintf(dest, size, format, ...)		snprintf((dest), (size), (format), __VA_ARGS__)
#define pub_vsnprintf(dest, size, format, va_ptr)	vsnprintf((dest), (size), (format), (va_ptr))

#endif

char*	pub_sprintf(const char* format, ...);
char*	pub_vsprintf(const char* format, va_list va_ptr);

#ifdef __cplusplus
}
#endif

#endif
