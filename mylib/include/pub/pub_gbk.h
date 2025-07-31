#ifndef	PUB_GBK_H_
#define PUB_GBK_H_

#ifdef __cplusplus
extern "C" {
#endif

enum GBKTYPE {PUB_GBK_NULL = 0, PUB_GBK_SPACE = 1, PUB_GBK_DIGIT = 2, PUB_GBK_ALPHA = 4,
	PUB_GBK_ENMARK = 8,	PUB_GBK_WORD = 16, PUB_GBK_CNMARK = 32, PUB_GBK_CNSIGN = 64};

int		pub_gbktype(const char* gbkstr);
int		pub_gbkchtype(const char* gbkch);
int		pub_gbklen(const char* gbkstr);
int		pub_gbkisword(const char* gbkch);
int		pub_gbkismark(const char* gbkch);
int		pub_gbkissign(const char* gbkch);
int		pub_gbkisrepeat(const char* gbkstr, int num);
int		pub_gbkisdust(const char* gbkstr, int level, int num);
int		pub_gbklike(const char* gbkstr1, const char* gbkstr2);
int		pub_gbkdist(const char* gbkstr1, const char* gbkstr2);
char*	pub_gbklower(char* gbkstr);
char*	pub_gbkupper(char* gbkstr);
char*	pub_gbknormal(char* gbkstr);
char*	pub_gbkncpy(char* dest, const char* src, size_t len);
char*	pub_gbkreplace(char* gbkstr, const char* set, char replace);
char*	pub_gbkcut(char* gbkstr, int type, char replace);
char*	pub_gbkpre(char* gbkstr, char** cur, char* swap);
char*	pub_gbksuf(char* gbkstr, char** cur);

#ifdef __cplusplus
}
#endif

#endif
