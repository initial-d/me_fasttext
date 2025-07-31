#ifndef SPELL_H_
#define SPELL_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPELL_SYLLABLE_DELIMS_OUT		('\'')
#define SPELL_SYLLABLE_DELIMS_IN		('')
#define SPELL_POLYPHONE_DELIMS			('|')
#define SPELL_POLYPHONE_PROB_DELIMS		(':')
#define spell_delete(spell)				{spell_delete_unsafe(spell);(spell) = NULL;}

typedef struct _Spell Spell;

Spell*	spell_new(void);
Spell*	spell_clone(Spell* spell);
void	spell_clear(Spell* spell);
void	spell_delete_unsafe(Spell* spell);
long	spell_store_to_stream(Spell* spell, FILE* stream);
Spell*	spell_load_from_stream(FILE* stream, size_t* o_pos);
long	spell_store_to_file(Spell* spell, const char* uri);
Spell*	spell_load_from_file(const char* uri);
void*	spell_map_to_buffer(Spell* spell, size_t* o_size);
Spell*	spell_map_from_buffer(void* buffer, size_t* o_pos);
int		spell_insert_syllable(Spell* spell, const char* uri);
int		spell_insert_sylpre(Spell* spell, const char* uri);
int		spell_insert_english(Spell* spell, const char* uri);
int		spell_insert_specabbr(Spell* spell, const char* uri);
int		spell_insert_word(Spell* spell, const char* word, const char* pinyin);
int		spell_insert_hztopy(Spell* spell, const char* uri);
int		spell_export_hztopy(Spell* spell, const char* uri);
int		spell_is_syllable(Spell* spell, const char* word);
int		spell_is_sylpre(Spell* spell, const char* word);
int		spell_is_pinyin(Spell* spell, const char* word);
int		spell_has_pinyin(Spell* spell, const char* word);
int		spell_is_english(Spell* spell, const char* word);
int		spell_has_english(Spell* spell, const char* word);
int		spell_is_specabbr(Spell* spell, const char* word);
int		spell_is_polyphone(Spell* spell, const char* hanzi);
int		spell_has_polyphone(Spell* spell, const char* word);
int		spell_is_match(Spell* spell, const char* word, const char* pinyin);
int		spell_split_pinyin(Spell* spell, const char* word, int do_tolerate, char* o_pinyin, size_t size);
int		spell_get_pinyin(Spell* spell, const char* word, int do_polyphone, char* o_pinyin, size_t size);
int		spell_get_pinyin_all(Spell* spell, const char* word, const char* delims, char* o_all, size_t size);
int		spell_get_pinyin_first(Spell* spell, const char* word, const char* delims, char* o_first, size_t size);
int		spell_get_pinyin_initials(Spell* spell, const char* word, const char* delims, char* o_initials, size_t size);
int		spell_translate_file(Spell* spell, const char* uri_in, const char* uri_out, const char* delims);
#ifndef _WIN32
Spell*	spell_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int		spell_munmap(void* mmap_addr, size_t mmap_len);
#endif

#ifdef __cplusplus
}
#endif

#endif
