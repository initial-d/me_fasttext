#ifndef PUB_ENCODE_H_
#define PUB_ENCODE_H_

#ifdef __cplusplus
extern "C" {
#endif

char*	pub_url_encode(const char* url_from, char* url_to, size_t size);
char*	pub_url_decode(const char* url_from, char* url_to, size_t size);
char*	pub_slashes_add(const char* src, const char* char_list, char* o_dest, size_t size);
char*	pub_slashes_del(const char* src, char* o_dest, size_t size);
char*	pub_entities_encode(const char* src, const char* char_list, char* o_dest, size_t size);
char*	pub_entities_decode(const char* src, char* o_dest, size_t size);

#ifdef __cplusplus
}
#endif

#endif
