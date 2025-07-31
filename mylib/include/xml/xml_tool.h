#ifndef XML_TOOL_H_
#define XML_TOOL_H_

#ifdef __cplusplus
extern "C" {
#endif

long	xml_xpath_get_long(const char* uri, const char* xpath);
double	xml_xpath_get_double(const char* uri, const char* xpath);
char*	xml_xpath_get_string(const char* uri, const char* xpath);

#ifdef __cplusplus
}
#endif

#endif
