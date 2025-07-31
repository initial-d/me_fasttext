#ifndef PUB_STRING_H_
#define PUB_STRING_H_

#include <ctype.h>
#include <string.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#define pub_isspace(ch)							(ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '\v' || ch == '\f')
#define pub_isalpha(ch)							((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
#define pub_isdigit(ch)							(ch >= '0' && ch <= '9')
#define pub_ismark(ch)							((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) || (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126))

#ifdef _WIN32

#define pub_strtok(str, delims, nouse_ptr)		strtok_s((str), (delims), &(nouse_ptr))
#define pub_strcasecmp							_stricmp
#define pub_strncasecmp							_strnicmp
#define pub_strdup								_strdup

#else

#define pub_strtok(str, delims, nouse_ptr)		strtok_r((str), (delims), &(nouse_ptr))
#define pub_strcasecmp							strcasecmp
#define pub_strncasecmp							strncasecmp
#define pub_strdup								strdup

#endif

char*	pub_strndup(const char* str, size_t len);
char*	pub_strncpy(char* dest, const char* src, size_t len);
char*	pub_strrev(char* str);
char*	pub_strltrim(char* str);
char*	pub_strrtrim(char* str);
char*	pub_stratrim(char* str);
char*	pub_strtrim(char* str);
int		pub_strstart(const char* str, const char* start);
int		pub_strend(const char* str, const char* end);
int		pub_strdist(const char* str1, const char* str2);

#ifdef __cplusplus
}
#endif

#endif
