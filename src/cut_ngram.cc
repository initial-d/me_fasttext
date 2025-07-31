#include "pub/pub.h"
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include "trie/trie.h"

using namespace std;


void computeNgrams(const std::string& word, std::vector<std::string>& substrings) {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= 6; n++) {
       ngram.push_back(word[j++]);
       while (j < word.size() && (word[j] & 0xC0) == 0x80) {
         ngram.push_back(word[j++]);
       }
       if (n >= 2 && !(n == 1 && (i == 0 || j == word.size()))) {
		 substrings.push_back(ngram);
	   }
	}
  }
}


int main(int argc, char* argv[])
{
    string line;
    ifstream fin(argv[1]);
    DATrie* cut_ngramdat = datrie_new();
    DATrie* ngramdat = datrie_load_from_file("ngramdat.bin");
	while (getline(fin, line))
	{
        string query = "<" + line + ">";
        vector<std::string> substrings;
        computeNgrams(query, substrings);
        for(int i = 0; i < substrings.size(); i++) {
            //cout << substrings[i] << endl;
	        size_t id = 0;
	        if (datrie_search_exact_match(ngramdat, substrings[i].c_str(), &id) == -1) continue;
		    datrie_insert_key(cut_ngramdat, substrings[i].c_str(), id);
        }
		//for (i = 0; i < 128; ++ i) printf("%f ", vec[i]);
		//printf("\n");
	}
    datrie_store_to_file(cut_ngramdat, "cut_ngramdat.bin");
	datrie_delete(ngramdat);
	datrie_delete(cut_ngramdat);

	return 0;
}

