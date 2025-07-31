#ifndef XML_PARSER_H_
#define XML_PARSER_H_

#include "xml_tree.h"

#ifdef __cplusplus
extern "C" {
#endif

Xml_Doc* xml_doc_parse(const char* uri);

#ifdef __cplusplus
}
#endif

#endif
