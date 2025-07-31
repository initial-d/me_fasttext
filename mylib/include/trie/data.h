#ifndef DATA_H_
#define DATA_H_

#include <stdio.h>
#include "trie/triemap.h"

#ifdef __cplusplus
extern "C" {
#endif

#define data_delete(data)								{data_delete_unsafe(data);(data) = NULL;}
#define data_fields_delete(data_fields)					{data_fields_delete_unsafe(data_fields);(data_fields) = NULL;}

typedef struct _Data Data;
typedef struct _Data_Fields Data_Fields;

enum DATA_TYPE {DATA_STRING = 0, DATA_LONG, DATA_ULONG, DATA_INT, DATA_UINT, DATA_SHORT, DATA_USHORT, DATA_CHAR,
	DATA_UCHAR, DATA_DOUBLE, DATA_FLOAT};

Data*			data_new(void);
Data*			data_clone(Data* data);
void			data_clear(Data* data);
void			data_delete_unsafe(Data* data);
long			data_store_to_stream(Data* data, FILE* stream);
Data*			data_load_from_stream(FILE* stream, size_t* o_pos);
long			data_store_to_file(Data* data, const char* uri);
Data*			data_load_from_file(const char* uri);
void*			data_map_to_buffer(Data* data, size_t* o_size);
Data*			data_map_from_buffer(void* buffer, size_t* o_pos);
int				data_insert(Data* data, Data_Fields* data_fields, TrieMap* map);
int				data_modify(Data* data, Data_Fields* data_fields, TrieMap* map, int id);
int				data_remove(Data* data, int id);
int				data_show(Data* data, const char* uri);
int				data_trim(Data* data);
int				data_get_maxid(Data* data);
int				data_get_field_type(Data* data, const char* field);
const char*		data_get_field_string(Data* data, int id, const char* field);
long			data_get_field_long(Data* data, int id, const char* field);
unsigned long	data_get_field_ulong(Data* data, int id, const char* field);
int				data_get_field_int(Data* data, int id, const char* field);
unsigned int	data_get_field_uint(Data* data, int id, const char* field);
short			data_get_field_short(Data* data, int id, const char* field);
unsigned short	data_get_field_ushort(Data* data, int id, const char* field);
char			data_get_field_char(Data* data, int id, const char* field);
unsigned char	data_get_field_uchar(Data* data, int id, const char* field);
double			data_get_field_double(Data* data, int id, const char* field);
float			data_get_field_float(Data* data, int id, const char* field);
int				data_format_custom(Data* data, int id, const char* format, const char* args, char* o_buf, size_t size);
int				data_format_tab(Data* data, int id, const char* args, char* o_buf, size_t size);
int				data_format_sxml(Data* data, int id, const char* args, char* o_buf, size_t size);
int				data_format_xml(Data* data, int id, const char* args, int indent, char* o_buf, size_t size);
int				data_format_json(Data* data, int id, const char* args, char* o_buf, size_t size);
int				data_format_ids(Data* data, int* ids, size_t ids_len, const char* format, const char* args,
				const char* begin_tag, const char* split_tag, const char* end_tag, char* o_buf, size_t size);
int				data_item_has_field(Data* data, int id, const char* field);
int				data_fields_init(Data* data, const char* fields);
Data_Fields*	data_fields_get(Data* data);
void			data_fields_delete_unsafe(Data_Fields* data_fields);

#ifdef __cplusplus
}
#endif

#endif
