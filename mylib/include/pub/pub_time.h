#ifndef PUB_TIME_H_
#define PUB_TIME_H_

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	int year;
	int month;
	int yday;
	int mday;
	int wday;
	int hour;
	int minute;
	int second;
	int msecond;
}Pub_Time;

Pub_Time	pub_time_local(void);
double		pub_time_clock(void);
void		pub_time_msleep(size_t msec);
long		pub_time_convert(const char* tstr);

#ifdef __cplusplus
}
#endif

#endif
