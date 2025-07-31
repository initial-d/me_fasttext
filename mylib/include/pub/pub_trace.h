#ifndef PUB_TRACE_H_
#define PUB_TRACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define PUB_DEBUG
#define PUB_ERROR

#define pub_debug(fmt, ...)				{pub_debug_brief("%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);\
										pub_debug_brief(fmt, __VA_ARGS__);}
#define pub_fdebug(uri, fmt, ...)		{pub_fdebug_brief(uri, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);\
										pub_fdebug_brief(uri, fmt, __VA_ARGS__);}
#define pub_error(fmt, ...)				{pub_error_brief("%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);\
										pub_error_brief(fmt, __VA_ARGS__);}
#define pub_ferror(uri, fmt, ...)		{pub_ferror_brief(uri, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__);\
										pub_ferror_brief(uri, fmt, __VA_ARGS__);}

void	pub_debug_brief(const char* fmt, ...);
void	pub_fdebug_brief(const char* uri, const char* fmt, ...);
void	pub_error_brief(const char* fmt, ...);
void	pub_ferror_brief(const char* uri, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif
