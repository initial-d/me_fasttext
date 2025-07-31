#ifndef PUB_FILE_H_
#define PUB_FILE_H_

#include <stdio.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define pub_fopen(uri, mode)			pub_fopen_detail((uri), (mode), __FILE__, __FUNCTION__, __LINE__)
#define pub_fclose(stream)				pub_fclose_detail((&(stream)), __FILE__, __FUNCTION__, __LINE__)
#define pub_rmdir(path)					rmdir(path)
#define pub_remove(uri)					pub_remove_detail((uri), __FILE__, __FUNCTION__, __LINE__)
#define pub_rename(uri_old, uri_new)	pub_rename_detail((uri_old), (uri_new), __FILE__, __FUNCTION__, __LINE__)
#define pub_access(uri, mode)			access((uri), (mode))

enum Mode{PUB_FILE_EXIST = 0, PUB_FILE_WRITE = 2, PUB_FILE_READ = 4, PUB_FILE_BOTH = 6};

FILE*	pub_fopen_detail(const char* uri, const char* mode, char* file_name, const char* function_name, int line_num);
int		pub_fclose_detail(FILE** stream, char* file_name, const char* function_name, int line_num);
int		pub_remove_detail(const char* uri, char* file_name, const char* function_name, int line_num);
int		pub_rename_detail(const char* uri_old, const char* uri_new, char* file_name, const char* function_name, int line_num);
char*	pub_fname(const char* uri);
char*	pub_fpath(const char* uri);
size_t	pub_fsize(const char* uri);
size_t	pub_fline(const char* uri);
int		pub_mkdir(const char* path);
int		pub_fchanged(const char* uri, int seconds);

#ifdef __cplusplus
}
#endif

#endif
