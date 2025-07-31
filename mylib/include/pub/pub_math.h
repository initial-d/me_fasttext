#ifndef PUB_MATH_H_
#define PUB_MATH_H_

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int pub_rand(unsigned int* seed);
int	pub_isprime(size_t num);

#ifdef __cplusplus
}
#endif

#endif
