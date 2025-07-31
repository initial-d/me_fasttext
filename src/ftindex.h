#pragma once
#include <string>
#include <vector>

typedef float real;
typedef struct _FTIndex FTIndex;

#define ftindex_delete(fti)        {ftindex_delete_unsafe(fti);(fti) = NULL;}

FTIndex*	ftindex_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len);
int			ftindex_munmap(void* mmap_addr, size_t mmap_len);
void		ftindex_delete_unsafe(FTIndex* fti);
int32_t		getWordId(FTIndex* fti, const std::string& word);
int32_t		getNgramId(FTIndex* fti, const std::string& word);
void		getWordVector(FTIndex* fti, const std::string& word, real* o_vec);
int		getWordVectorCnt(FTIndex* fti, const std::string& word, real* o_vec);
void		getNgramVector(FTIndex* fti, const std::string& word, real* o_vec);
void		getSentenceVector(FTIndex* fti, const std::string& sentence, real* o_vec);
void		getSentenceVector(FTIndex* fti, const std::string& sentence, real* o_vec, float weight);
void  		getPosBiasVector(FTIndex* fti, const std::string& sentence, real* o_vec, float weight);
void		computeSubwords(FTIndex*, const std::string& word,
			std::vector<uint32_t>& ids, 
			std::vector<std::string>& subwords);
void		computeSubwords(FTIndex* fti, const std::string& word,
			std::vector<std::string>& subwords,
			int32_t len);
