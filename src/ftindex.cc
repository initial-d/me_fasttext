#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "pub/pub.h"
#include "trie/trie.h"
#include "fasttext.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "ftindex.h"

using namespace std;
using namespace fasttext;

#define BOW								('<')
#define EOW								('>')
#define hash_new(h)						{(h) = 2166136261;}
#define hash_step(h, ch)				{(h) ^= uint32_t(ch);(h) *= 16777619;}

struct _FTIndex
{
	int32_t		nwords;
	int32_t		nngrams;
	int32_t		dim;
	int32_t		minn;
	int32_t		maxn;
	real*		data;
	DASTrie*	words2id;
	DASTrie*	ngrams2id;
};

real get_vector_norm(real* vec, int32_t dim)
{
	real sum = 0;

	for (int32_t i = 0; i < dim; ++ i)
	{
		real val = *(vec + i);
		sum += pow(val, 2) ;
		//sum += *(vec + i) * *(vec + i);
	}
	if(sum < 1e-6)
		return 0;
	return sqrt(sum);
}

uint32_t hash(const std::string& word)
{
	uint32_t h = 0, i = 0;

	hash_new(h);
	while (word[i])
	{
		hash_step(h, word[i ++]);
	}

	return h;
}

int32_t getWordId(FTIndex* fti, const std::string& word)
{
	size_t id = 0;

	if (dastrie_search_exact_match(fti->words2id, word.c_str(), &id) == -1) return -1;

	return (int32_t)id;
}

int32_t getNgramId(FTIndex* fti, const std::string& word)
{
	size_t id = 0;

	if (dastrie_search_exact_match(fti->ngrams2id, word.c_str(), &id) == -1) return -1;

	return (int32_t)id;
}

void computeSubwords(FTIndex* fti, const std::string& word,
		std::vector<uint32_t>& ids)
{
	ids.clear();
	std::string subword;
	for (size_t i = 0; i < word.size(); ++ i)
	{
		if ((word[i] & 0xC0) == 0x80) continue;
		subword.clear();
		for (size_t j = i, n = 1; j < word.size() && n <= fti->maxn; ++ n)
		{
            subword.push_back(word[j++]);
			while (j < word.size() && (word[j] & 0xC0) == 0x80)
			{
				subword.push_back(word[j++]);
			}
			if (n >= fti->minn && !(n == 1 && (i == 0 || j == word.size())))
			{
                size_t id = 0;
                if (dastrie_search_exact_match(fti->ngrams2id, subword.c_str(), &id) != -1)
                {
				    ids.push_back((int32_t)id + fti->nwords);
                }
			}
		}
	}
}

void computeSubwords(FTIndex* fti, const std::string& word,
		std::vector<uint32_t>& ids,
		std::vector<std::string>& subwords)
{
	ids.clear();
	subwords.clear();
	std::string subword;
	for (size_t i = 0; i < word.size(); ++ i)
	{
		if ((word[i] & 0xC0) == 0x80) continue;
		subword.clear();
		for (size_t j = i, n = 1; j < word.size() && n <= fti->maxn; ++ n)
		{
            subword.push_back(word[j++]);
			while (j < word.size() && (word[j] & 0xC0) == 0x80)
			{
				subword.push_back(word[j++]);
			}
			if (n >= fti->minn && !(n == 1 && (i == 0 || j == word.size())))
			{
                size_t id = 0;
                if (dastrie_search_exact_match(fti->ngrams2id, subword.c_str(), &id) != -1)
                {
				    subwords.push_back(subword);
				    ids.push_back((int32_t)id + fti->nwords);
                }
			}
		}
	}
}

void computeSubwords(FTIndex* fti, const std::string& word,
		std::vector<std::string>& subwords,
		int32_t len)
{
	subwords.clear();
	std::string subword;
	for (size_t i = 0; i < word.size(); ++ i)
	{
		if ((word[i] & 0xC0) == 0x80) continue;
		subword.clear();
		for (size_t j = i, n = 1; j < word.size() && n <= len; ++ n)
		{
			subword.push_back(word[j++]);
			while (j < word.size() && (word[j] & 0xC0) == 0x80)
			{
				subword.push_back(word[j++]);
			}
			if (n == len)
			{
				subwords.push_back(subword);
			}
		}
	}
}

void getWordVector(FTIndex* fti, const std::string& word, real* o_vec)
{
	uint32_t i = 0, j = 0;
	size_t id = 0;
	real* cur = NULL;
    
	if (dastrie_search_exact_match(fti->words2id, word.c_str(), &id) == 0)
	{
		memcpy(o_vec, fti->data + fti->dim * id, fti->dim * sizeof(*o_vec));
        //for(int k = 0; k < fti->dim; k++) {
        //    std::cout << *(fti->data + fti->dim * id + k) << " ";
        //}
        //std::cout << endl;
		return;
	}
	vector<uint32_t> ids;
	ids.reserve(1024);
	computeSubwords(fti, BOW + word + EOW, ids);
	memset(o_vec, 0, fti->dim * sizeof(*o_vec));
	for (i = 0; i < ids.size(); ++ i)
	{
		cur = fti->data + (size_t)fti->dim * ids[i];
		for (j = 0; j < fti->dim; ++ j)
		{
			o_vec[j] += cur[j];
		}
	}
	if (ids.size() != 0)
	{
		for (i = 0; i < fti->dim; ++ i) o_vec[i] /= ids.size();
	}
}

int getWordVectorCnt(FTIndex* fti, const std::string& word, real* o_vec)
{
	uint32_t i = 0, j = 0;
	size_t id = 0;
	real* cur = NULL;

	vector<uint32_t> ids;
	ids.reserve(1024);
	computeSubwords(fti, BOW + word + EOW, ids);
	memset(o_vec, 0, fti->dim * sizeof(*o_vec));
	for (i = 0; i < ids.size(); ++ i)
	{
		cur = fti->data + (size_t)fti->dim * ids[i];
		for (j = 0; j < fti->dim; ++ j)
		{
			o_vec[j] += cur[j];
		}
	}

    
	if (dastrie_search_exact_match(fti->words2id, word.c_str(), &id) == 0)
	{
		memcpy(o_vec, fti->data + fti->dim * id, fti->dim * sizeof(*o_vec));
		if(word.find("p#") == 0) {
			return 1;
		}
		for(int k = 0; k < fti->dim; k++) {
			o_vec[k] *= (ids.size() + 1);
		}
		return ids.size() + 1;
	}
	return ids.size();
}

void getNgramVector(FTIndex* fti, const std::string& word, real* o_vec)
{
	uint32_t j = 0;
	size_t id = 0;
	real* cur = NULL;
    
	memset(o_vec, 0, fti->dim * sizeof(*o_vec));
	id = getNgramId(fti, word);
	
    if(id == -1){
		return;
	}
	cur = fti->data + (size_t)fti->dim * (id + fti->nwords);
	for (j = 0; j < fti->dim; ++ j)
	{
		o_vec[j] += cur[j];
	}
}

void getSentenceVector(FTIndex* fti, const std::string& sentence, real* o_vec)
{
	istringstream iss(sentence);
	string word;
	int32_t count = 0, i = 0;
	real norm = 0, vec[128] = {0};
	//real norm = 0, *vec = NULL;


	memset(o_vec, 0, fti->dim * sizeof(*o_vec));
	//if (!(vec = (real*)pub_calloc(fti->dim, sizeof(*vec)))) return;
	while (iss >> word)
	{
		getWordVector(fti, word, vec);
		norm = get_vector_norm(vec, fti->dim);
		//norm = 1;
		if (norm <= 0) continue;
		for (i = 0; i < fti->dim; ++ i)
		{
			o_vec[i] += vec[i] / norm;
		}
		++ count;
	}
	if (count > 1)
	{
		for (i = 0; i < fti->dim; ++ i)
		{
			o_vec[i] /= count;
		}
	}
	//pub_free(vec);
}

void getSentenceVector(FTIndex* fti, const std::string& sentence, real* o_vec, float weight)
{
	istringstream iss(sentence);
	string word;
	int32_t count = 0, i = 0;
	real norm = 0, vec[128] = {0};
	//real norm = 0, *vec = NULL;

	memset(o_vec, 0, fti->dim * sizeof(*o_vec));
	//if (!(vec = (real*)pub_calloc(fti->dim, sizeof(*vec)))) return;
	while (iss >> word)
	{
		bool flag = false;
		if(word.find("p#") == 0) {flag = true;}
		getWordVector(fti, word, vec);

		norm = get_vector_norm(vec, fti->dim);
		//norm = 1;
		if (norm <= 0) continue;
		for (i = 0; i < fti->dim; ++ i)
		{
			if(flag) {
				o_vec[i] += (vec[i] / norm) * weight;
			}
			else {
				o_vec[i] += vec[i] / norm;
			}
		}
		++ count;
	}
	if (count > 1)
	{
		for (i = 0; i < fti->dim; ++ i)
		{
			o_vec[i] /= count;
		}
	}

	//pub_free(vec);
}

void getPosBiasVector(FTIndex* fti, const std::string& sentence, real* o_vec, float weight)
{
	istringstream iss(sentence);
	string word;
	int32_t count = 0, i = 0;
	real norm = 0, vec[128] = {0};
	//real norm = 0, *vec = NULL;

	memset(o_vec, 0, fti->dim * sizeof(*o_vec));
	//if (!(vec = (real*)pub_calloc(fti->dim, sizeof(*vec)))) return;
	int cnt = 0;
	while (iss >> word)
	{
		bool flag = false;
		if(word.find("p#") == 0) {flag = true;}
		int n_size = getWordVectorCnt(fti, word, vec);
		cnt += n_size;

		//norm = get_vector_norm(vec, fti->dim);
		//norm = 1;
		//if (norm <= 0) continue;
		for (i = 0; i < fti->dim; ++ i)
		{
			if(flag) {
				//o_vec[i] += (vec[i] / norm) * weight;
				//o_vec[i] += vec[i] / norm;
				o_vec[i] += vec[i] * weight;
			}
			else {
				o_vec[i] += vec[i];
			}
		}
		++ count;
	}
	if (cnt > 1)
	{
		for (i = 0; i < fti->dim; ++ i)
		{
			o_vec[i] /= cnt;
		}
	}

	//pub_free(vec);
}

#if 0
void getPosBiasVector(FTIndex* fti, const std::string& sentence, real* o_vec, float weight)
{
	istringstream iss(sentence);
	string word;
	int32_t count = 0, i = 0;
	real norm = 0, vec[128] = {0};
	//real norm = 0, *vec = NULL;
	real pos_vec[128] = {0};
	real q_vec[128] = {0};

	memset(o_vec, 0, fti->dim * sizeof(*o_vec));
	//if (!(vec = (real*)pub_calloc(fti->dim, sizeof(*vec)))) return;
	int cnt = 0;
	while (iss >> word)
	{
		bool flag = false;
		int n_size = getWordVectorCnt(fti, word, vec);
		cnt += n_size;
		if(word.find("p#") == 0) {
			for(int i = 0; i < fti->dim; i++) {
				pos_vec[i] = vec[i];
			}
		}
		else {
			for(int i = 0; i < fti->dim; i++) {
				q_vec[i] += vec[i];
			}
		}

		//norm = get_vector_norm(vec, fti->dim);
		//norm = 1;
		//if (norm <= 0) continue;
	}
	
	float pos_norm = get_vector_norm(pos_vec, fti->dim);
	float q_norm = get_vector_norm(q_vec, fti->dim);
	float cos = 0.0;
	float dot = 0.0;
	for(int i = 0; i < fti->dim; i++) {
		dot += pos_vec[i] * q_vec[i];
	}
	cos = dot / (pos_norm * q_norm);
	
	for (i = 0; i < fti->dim; ++ i) {
		o_vec[i] = q_vec[i] + cos * pos_vec[i];
	}

	if (cnt > 1)
	{
		for (i = 0; i < fti->dim; ++ i)
		{
			o_vec[i] /= cnt;
		}
	}

	//pub_free(vec);
}
#endif

void ftindex_delete_unsafe(FTIndex* fti)
{
	dastrie_delete(fti->words2id);
	pub_free(fti);
}

FTIndex* ftindex_map_from_buffer(void* buffer, size_t* o_pos)
{
	FTIndex* fti = NULL;
	char* cur = NULL;
	size_t size = 0;

	if (!buffer || !o_pos) return NULL;
	if (!(fti = (FTIndex*)pub_calloc(1, sizeof(*fti)))) return NULL;
	cur = (char*)buffer + *o_pos;
	fti->nwords = *(int32_t*)cur;
	cur += sizeof(fti->nwords);
	fti->nngrams = *(int32_t*)cur;
	cur += sizeof(fti->nngrams);
	fti->dim = *(int32_t*)cur;
	cur += sizeof(fti->dim);
	fti->minn = *(int32_t*)cur;
	cur += sizeof(fti->minn);
	fti->maxn = *(int32_t*)cur;
	cur += sizeof(fti->maxn);
	fti->data = (real*)cur;
	cur += sizeof(*fti->data) * fti->dim * (fti->nwords + fti->nngrams);
	if (!(fti->words2id = dastrie_map_from_buffer(cur, &size))) {
        dastrie_delete(fti->words2id);
	    pub_free(fti);
	    return NULL;
    }
	cur += size;
    size = 0;
	if (!(fti->ngrams2id = dastrie_map_from_buffer(cur, &size))) {
        dastrie_delete(fti->words2id);
        dastrie_delete(fti->ngrams2id);
	    pub_free(fti);
	    return NULL;
    }
	cur += size;

    size_t id = 0;

     //for test
    /*
    std::string s = "爱奇艺";
    if (dastrie_search_exact_match(fti->words2id, s.c_str(), &id) != -1) {
        std::cout << "爱奇艺 " << id << endl;
    }
    std::cout << "nngrams_size: " << (size_t)dastrie_get_key_num(fti->ngrams2id) << endl;
    std::cout << "nngrams_size: " << (size_t)size << endl;
    if (dastrie_search_exact_match(fti->ngrams2id, s.c_str(), &id) != -1) {
        std::cout << "爱奇艺 " << id << endl;
    }
    for(int k = 0; k < fti->dim; k++) {
        std::cout << *(fti->data + fti->dim * id + k) << " ";
    }
    std::cout<<std::endl;
    */


	*o_pos = cur - (char*)buffer;
	return fti;


}

FTIndex* ftindex_mmap(const char* uri, void** o_mmap_addr, size_t* o_mmap_len)
{
	size_t pos = 0;
	int fd = 0;
	struct stat st;
	FTIndex* fti = NULL;

	if (!uri || !o_mmap_addr || !o_mmap_len) return NULL;
	if ((fd = open(uri, O_RDONLY)) == -1) goto gt;
	if (fstat(fd, &st) == -1) goto gt;
	*o_mmap_len = st.st_size;
	if ((*o_mmap_addr = mmap(NULL, *o_mmap_len, PROT_READ, MAP_POPULATE|MAP_SHARED, fd, 0)) == (void*)(-1)) goto gt;
	if (!(fti = ftindex_map_from_buffer(*o_mmap_addr, &pos))) goto gt;

gt:	close(fd);
	return fti;
}

int ftindex_munmap(void* mmap_addr, size_t mmap_len)
{
	if (!mmap_addr) return -1;
	if (munmap(mmap_addr, mmap_len) == -1) return -1;

	return 0;
}
