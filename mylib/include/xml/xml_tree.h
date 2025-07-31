#ifndef XML_TREE_H_
#define XML_TREE_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define xml_doc_delete(doc)					{xml_doc_delete_unsafe(doc); (doc) = NULL;}
#define xml_node_delete(node)				{xml_node_delete_unsafe(node); (node) = NULL;}
#define xml_node_list_delete(node)			{xml_node_list_delete_unsafe(node); (node) = NULL;}

typedef struct _Xml_Doc Xml_Doc;
typedef struct _Xml_Node Xml_Node;
typedef struct _Xml_Prop Xml_Prop;

Xml_Doc*	xml_doc_new(const char* version);
void		xml_doc_delete_unsafe(Xml_Doc* doc);
void		xml_doc_dump(FILE* stream, Xml_Doc* doc);
void		xml_doc_save(const char* uri, Xml_Doc* doc);
void		xml_doc_set_root(Xml_Doc* doc, Xml_Node* root);
Xml_Node*	xml_doc_get_root(Xml_Doc* doc);
Xml_Node*	xml_node_new(const char* name, const char* content);
void		xml_node_unlink(Xml_Node* node);
void		xml_node_delete_unsafe(Xml_Node* node);
void		xml_node_list_delete_unsafe(Xml_Node* node);
void		xml_node_dump(FILE* stream, Xml_Node* node, int level);
void		xml_node_list_dump(FILE* stream, Xml_Node* node, int level);
Xml_Node*	xml_node_new_child(Xml_Node* parent, const char* name, const char* content);
Xml_Node*	xml_node_add_child(Xml_Node* parent, Xml_Node* child);
Xml_Node*	xml_node_get_child(Xml_Node* parent, const char* name);
void		xml_node_delete_child(Xml_Node*	parent, const char* name);
Xml_Node*	xml_node_get_first_child(Xml_Node* parent);
Xml_Node*	xml_node_get_last_child(Xml_Node* parent);
Xml_Node*	xml_node_get_next_sibling(Xml_Node* node);
Xml_Node*	xml_node_get_prev_sibling(Xml_Node* node);
void		xml_node_set_name(Xml_Node* node, const char* name);
const char*	xml_node_get_name(Xml_Node* node);
int			xml_node_get_count(Xml_Node* node);
void		xml_node_set_content(Xml_Node* node, const char* content);
const char*	xml_node_get_content(Xml_Node* node);
Xml_Prop*	xml_node_has_prop(Xml_Node* node, const char* name);
Xml_Prop*	xml_node_set_prop(Xml_Node* node, const char* name, const char* value);
void		xml_node_unset_prop(Xml_Node* node, const char* name);
const char*	xml_node_get_prop(Xml_Node* node, const char* name);

#ifdef __cplusplus
}
#endif

#endif
