#include "pub/pub.h"
#include "fasttext.h"
#include "ftindex.h"
#include <iostream>
#include <sstream>
#include <string>
#include "trie/trie.h"

using namespace std;
using namespace fasttext;

int main(int argc, char* argv[])
{
	FTIndex* fti = NULL;
	void* mmap_addr = NULL;
	size_t mmap_len = 0;
	int i = 0, j = 0;
	string line;
	real vec[128] = {0};
	double start = 0, end = 0;

	{
	  //FastText fasttext;
	  //fasttext.loadModel(std::string(argv[1]));
	  //fasttext.saveIndex(argv[1]);
	  //fasttext.saveOutput();
      DASTrie* cut_ngramdat = dastrie_new();
      DASTrie* ngramdat = dastrie_load_from_file("ngramdat.bin");

      std::cout << "nngrams_size: " << (int32_t)dastrie_get_key_num(ngramdat) << std::endl;
      //int64_t nums = fasttext.getArgs().wordNgrams;
        //cout << "ngram: " << nums << endl;
    }
	//fti = ftindex_mmap(argv[1], &mmap_addr, &mmap_len);
	//while (getline(cin, line))
	//{
	//	if (line == "exit") break;
	//	start = pub_time_clock();
	//	for (i = 0; i < 1000000; ++ i)
	//	{
	//		getSentenceVector(fti, line, vec);
	//	}
	//	end = pub_time_clock();
	//	printf("search time: %lf\n", end - start);
	//	for (i = 0; i < 128; ++ i) printf("%f ", vec[i]);
	//	printf("\n");
	//}
	//ftindex_munmap(mmap_addr, mmap_len);
	//ftindex_delete(fti);

	return 0;
}

