/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <assert.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <stdexcept>

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";
size_t ncount = 0;

void Dictionary::getNgrams_is(const std::string& word,
		std::vector<int32_t>& ngrams,
		std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(1);
    substrings.push_back(words_[i].word);
  } else {
    ngrams.push_back(-1);
    substrings.push_back(word);
  }
  computeNgrams_is(word, ngrams, substrings);
}

void Dictionary::getNgrams_is_2(const std::string& word,
                           std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  substrings.clear();
  if (i >= 0) {
    substrings.push_back(words_[i].word);
  } 
  //computeNgrams(BOW + word + EOW, ngrams, substrings);
  computeNgrams_is_2(word, substrings);
}

void Dictionary::computeNgrams_is(const std::string& word,
                              std::vector<int32_t>& ngrams,
                              std::vector<std::string>& substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
   std::string ngram;
   if ((word[i] & 0xC0) == 0x80) continue;
   for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
     ngram.push_back(word[j++]);
     while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
     }
     if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
     //if (n >= 2 && !(n == 1 && (i == 0 || j == word.size()))) {
        //int32_t h = hash(ngram) % args_->bucket;
		int32_t i = getId(ngram);
		if (i >= 0) {
			ngrams.push_back(1);
			substrings.push_back(ngram);
		}
		else {
			ngrams.push_back(-1);
			substrings.push_back(ngram);
		}
	 }
	}
}
}

void Dictionary::computeNgrams_is_2(const std::string& word,
                              std::vector<std::string>& substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= 2; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      //if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
      if (n >= 2 && !(n == 1 && (i == 0 || j == word.size()))) {
      //int32_t h = hash(ngram) % args_->bucket;
		int32_t i = getId(ngram);
		if (i >= 0) {
			substrings.push_back(ngram);
		}
	  }
	}
  }
}

Dictionary::Dictionary(std::shared_ptr<Args> args) : args_(args),
  word2int_(MAX_VOCAB_SIZE, -1), size_(0), nwords_(0), nngrams_(0), nlabels_(0),
  ntokens_(0), pruneidx_size_(-1), worddat(NULL), ngramdat(NULL){}

Dictionary::~Dictionary()
{
  datrie_delete(worddat);
  datrie_delete(ngramdat);
}

int32_t Dictionary::find(const std::string& w) const {
  /*return find(w, hash(w));*/
	size_t id = 0;
	if (datrie_search_exact_match(worddat, w.c_str(), &id) == -1) return -1;
	return (int32_t)id;

}

/*int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t id = h % MAX_VOCAB_SIZE;
  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
    id = (id + 1) % MAX_VOCAB_SIZE;
  }
  return id;
}*/

void Dictionary::add(const std::string& w) {
  int32_t h = find(w);
 
  bool src_flag = false;
  if(w.find("s#") == 0) {
	src_flag = true;
  }
  if(!src_flag) {
  	ntokens_++;
  }
  /*if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }*/
  if (h == -1) {
    entry e;
    e.word = w;
    if(src_flag) {
    	e.count = 2147483647;
    } else {
    	e.count = 1;
    }
    e.type = getType(w);
    words_.push_back(e);
    if (datrie_insert_key(worddat, w.c_str(), size_++) == -1)
		std::cout << "worddat insert error: " << w << std::endl;
  } else {
    if(src_flag) {
    	words_[h].count = 2147483647;
    } else {
    	words_[h].count++;
    }
  }

}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nngrams() const {
  return nngrams_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const{
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getSubwords(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwords2(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

void Dictionary::getSubwords(const std::string& word,
                           std::vector<int32_t>& ngrams,
                           std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwords2(BOW + word + EOW, ngrams, substrings);
  }
}

bool Dictionary::discard(int32_t id, real rand) const { //???
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  /*int32_t id = find(w, h);
  return word2int_[id];*/
	return find(w);
}

int32_t Dictionary::getId(const std::string& w) const {
  /*int32_t h = find(w);
  return word2int_[h];*/
	return find(w);
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeSubwords(const std::string& word,
                               std::vector<int32_t>& ngrams,
                               std::vector<std::string>& substrings) const{
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        /*int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);*/
		size_t id = 0;
		if (datrie_search_exact_match(ngramdat, ngram.c_str(), &id) == -1)
		{
			if (datrie_insert_key(ngramdat, ngram.c_str(), ncount) == -1)
				std::cout << "insert ngram error: " << ngram << " " << ncount << std::endl;
			id = ncount++;
		}
		ngrams.push_back(nwords_ + (int32_t)id);

        substrings.push_back(ngram);
      }
    }
  }
}

void Dictionary::computeSubwords(const std::string& word,
                               std::vector<int32_t>& ngrams) const{
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        /*int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);*/
		size_t id = 0;
		if (datrie_search_exact_match(ngramdat, ngram.c_str(), &id) == -1)
		{
			if (datrie_insert_key(ngramdat, ngram.c_str(), ncount) == -1)
				std::cout << "insert ngram error: " << ngram << " " << ncount << std::endl;
			id = ncount++;
		}
		ngrams.push_back(nwords_ + (int32_t)id);

      }
    }
  }
}

void Dictionary::computeSubwords2(const std::string& word,
                               std::vector<int32_t>& ngrams,
                               std::vector<std::string>& substrings) const{
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        /*int32_t h = hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);*/
		size_t id = 0;
		if (datrie_search_exact_match(ngramdat, ngram.c_str(), &id) != -1)
		{
			ngrams.push_back(nwords_ + (int32_t)id);
	        substrings.push_back(ngram);
		}
      }
    }
  }
}

void Dictionary::computeSubwords2(const std::string& word,
                               std::vector<int32_t>& ngrams) const{
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        /*int32_t h = hash(ngram) % args_->bucket;
        pushHash(ngrams, h);*/
		size_t id = 0;
		if (datrie_search_exact_match(ngramdat, ngram.c_str(), &id) != -1)
		{
			ngrams.push_back(nwords_ + (int32_t)id);
		}

      }
    }
  }
}

void Dictionary::initNgrams() {
	ncount = 0;
  for (size_t i = 0; i < size_; i++) {
    if(words_[i].word.find("p#") == 0) {
	continue;
    }
    if(words_[i].word.find("s#") == 0) {
	continue;
    }
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
  nngrams_ = (int32_t)datrie_get_key_num(ngramdat);
}

bool Dictionary::readWord(std::istream& in, std::string& word) const
{
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;

  worddat = datrie_new();
  ngramdat = datrie_new();
  while (readWord(in, word)) {
    //replace(word.begin(), word.end(), ' ', '');
    std::string query;
    std::string src;
    if(word.find("p#") == 0 || word == EOS) {
    	query = word;
	add(query);
    }   
    else if(word.find("\x01") != std::string::npos) {
	query = word.substr(0, word.find("\x01"));
	src = word.substr(word.find("\x01") + 1);
	if(query == "" || src == "") {
	    continue;
	} 
        add(query);
        add("s#" + src);
    } else {
    	continue;
    }
 
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
    }
    /*if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold, minThreshold);
    }*/
  }
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.count > e2.count;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
        return (e.type == entry_type::word && e.count < t) ||
               (e.type == entry_type::label && e.count < tl);
      }), words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nngrams_ = 0;
  nlabels_ = 0;
  /*std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }*/
  datrie_clear(worddat);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
	if (datrie_insert_key(worddat, it->word.c_str(), size_++) == -1)
		std::cout << "worddat trie insert error: " << it->word << std::endl;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }

}

void Dictionary::initTableDiscard() {//???
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) counts.push_back(w.count);
  }
  return counts;
}

void Dictionary::addWordNgrams(std::vector<int32_t>& line,
                               const std::vector<int32_t>& hashes,
                               int32_t n) const {
  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(std::vector<int32_t>& line,
                             const std::string& token,
                             int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      computeSubwords(BOW + token + EOW, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  while (readWord(in, token)) {
    /*int32_t h = find(token);
    int32_t wid = word2int_[h];*/
    int32_t wid = find(token);
    //std::cout << token << " " << wid << std::endl;

    if (wid < 0) continue;

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) break;
  }
  //std::cout << ntokens << std::endl;
  return ntokens;
}

int32_t Dictionary::getLine_pos_src(std::istream& in,
                            std::vector<int32_t>& words,
			    std::vector<int32_t>& srcs,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  srcs.clear();
  while (readWord(in, token)) {
    /*int32_t h = find(token);
    int32_t wid = word2int_[h];*/


    std::string query = "";
    std::string src = "";
    int32_t wid = 0, sid = 0;
    if(token.find("p#") != std::string::npos || token == EOS) {
	query = token;
    	wid = find(query);
	sid = -1;
    }
    else if(token.find("\x01") != std::string::npos) {
	query = token.substr(0, token.find("\x01"));
	src = token.substr(token.find("\x01") + 1);
    	if(query == "" || src == "") {
	    continue;
    	} 
	wid = find(query);
	sid = find("s#" + src);
    } else {
   	continue; 	
    }
   
    //std::cout << token << std::endl;
    //std::cout << query << "#" << src << "#" << wid << "#" << sid << std::endl;

    //int32_t wid = find(query);
    //int32_t sid = find(src);
    //std::cout << token << " " << wid << std::endl;

    if (wid < 0) continue;

    ntokens++;
    
    if(getType(wid) == entry_type::word && token.find("p#") != std::string::npos) {
      words.push_back(wid);
      srcs.push_back(sid);
    }
    else if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
      srcs.push_back(sid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) break;
  }
  //std::cout << ntokens << std::endl;
  return ntokens;
}

int32_t Dictionary::getLine_pos_ads(std::istream& in,
                            std::vector<int32_t>& words,
			    std::vector<int32_t>& ads,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  ads.clear();
  while (readWord(in, token)) {
    if(token.find("ad#") == 0) {
	int32_t aid = find(token.substr(3));
	if(aid < 0) {continue;}
    	ntokens++;
	ads.push_back(aid);
	continue;
    }
    /*int32_t h = find(token);
    int32_t wid = word2int_[h];*/
    int32_t wid = find(token);
    //std::cout << token << " " << wid << std::endl;

    if (wid < 0) continue;

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) break;
  }
  //std::cout << ntokens << std::endl;
  return ntokens;
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::vector<int32_t>& labels,
                            std::minstd_rand& rng) const {
  std::vector<int32_t> word_hashes;
  std::string token;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  labels.clear();
  while (readWord(in, token)) {
    uint32_t h = hash(token);
    int32_t wid = getId(token, h);
    entry_type type = wid < 0 ? getType(token) : getType(wid);

    ntokens++;
    if (type == entry_type::word) {
      addSubwords(words, token, wid);//将所有token的char ngram, 以及token加入words
      word_hashes.push_back(h);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(wid - nwords_);
    }
    if (token == EOS) break;
  }
  addWordNgrams(words, word_hashes, args_->wordNgrams);//将所有token的word ngram加入words
  return ntokens;
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) return;
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  datrie_store_to_file(worddat, "worddat.bin");
  datrie_store_to_file(ngramdat, "ngramdat.bin");
  out.write((char*) &size_, sizeof(int32_t));
  out.write((char*) &nwords_, sizeof(int32_t));
  out.write((char*) &nngrams_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &ntokens_, sizeof(int64_t));
  out.write((char*) &pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*) &(e.count), sizeof(int64_t));
    out.write((char*) &(e.type), sizeof(entry_type));
  }
  for (const auto pair : pruneidx_) {
    out.write((char*) &(pair.first), sizeof(int32_t));
    out.write((char*) &(pair.second), sizeof(int32_t));
  }
}

void Dictionary::load(std::istream& in) {
  //worddat = datrie_load_from_file("worddat.bin");
  //ngramdat = datrie_load_from_file("ngramdat.bin");
  worddat = datrie_new();
  ngramdat = datrie_new();
  words_.clear();
  std::fill(word2int_.begin(), word2int_.end(), -1);
  in.read((char*) &size_, sizeof(int32_t));
  in.read((char*) &nwords_, sizeof(int32_t));
  in.read((char*) &nngrams_, sizeof(int32_t));
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &ntokens_, sizeof(int64_t));
  in.read((char*) &pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*) &e.count, sizeof(int64_t));
    in.read((char*) &e.type, sizeof(entry_type));
    words_.push_back(e);
    //word2int_[find(e.word)] = i;
    if (datrie_insert_key(worddat, e.word.c_str(), i) == -1)
		std::cout << "worddat insert error: " << e.word << std::endl;
  }
  pruneidx_.clear();
  for (int32_t i = 0; i < pruneidx_size_; i++) {
    int32_t first;
    int32_t second;
    in.read((char*) &first, sizeof(int32_t));
    in.read((char*) &second, sizeof(int32_t));
    pruneidx_[first] = second;
  }
  initTableDiscard();
  initNgrams();
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {words.push_back(*it);}
    else {ngrams.push_back(*it);}
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  for (int32_t i = 0; i < words_.size(); i++) {
    if (getType(i) == entry_type::label || (j < words.size() && words[j] == i)) {
      words_[j] = words_[i];
      word2int_[find(words_[j].word)] = j;
      j++;
    }
  }
  nwords_ = words.size();
  size_ = nwords_ +  nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

}
