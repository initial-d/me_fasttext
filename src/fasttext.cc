/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"
#include "pub/pub.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <numeric>


namespace fasttext {

using namespace std;
constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

FastText::FastText() : quant_(false) {}

void FastText::addInputVector(Vector& vec, int32_t ind) const {
  if (quant_) {
    vec.addRow(*qinput_, ind);
  } else {
    vec.addRow(*input_, ind);
  }
}

std::shared_ptr<const Dictionary> FastText::getDictionary() const {
  return dict_;
}

const Args FastText::getArgs() const {
  return *args_.get();
}

std::shared_ptr<const Matrix> FastText::getInputMatrix() const {
  return input_;
}

std::shared_ptr<const Matrix> FastText::getOutputMatrix() const {
  return output_;
}

int32_t FastText::getWordId(const std::string& word) const {
  return dict_->getId(word);
}

int32_t FastText::getSubwordId(const std::string& word) const {
  int32_t h = dict_->hash(word) % args_->bucket;
  return dict_->nwords() + h;
}

void FastText::getVectorNorm(Vector& vec, const std::string& word) {
	const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
	vec.zero();
	for (auto it = ngrams.begin(); it != ngrams.end(); ++it)
	{
		//vec.addRowNorm(*input_, *it);
	}
	real norm = vec.norm();
	if(norm < 10e-13)
	{
		return;
	}
	if (ngrams.size() > 0) {
		vec.mul(1.0 / norm);
	}
}

void FastText::wordVectors(const std::string& word, std::string& res) {
	Vector vec(args_->dim);
	getWordVector(vec, word);
	std::ostringstream oss;
	oss << vec;
	res = oss.str();
}

void FastText::sentenceVectors(const std::string& sentence, std::string& res) {
	Vector vec(args_->dim);
	Vector svec(args_->dim);
	std::string word;
	std::istringstream iss(sentence);
	svec.zero();
	int32_t count = 0;
	while(iss >> word) {
		getWordVector(vec, word);
		vec.mul(1.0 / vec.norm());
		svec.addVector(vec);
		count++;
	}
	svec.mul(1.0 / count);
	std::ostringstream oss;
	oss << svec;
	res = oss.str();
}

void FastText::ngramVectors(std::string word, std::vector<std::string>& sub_vec) {
	std::vector<int32_t> ngrams;
	std::vector<std::string> substrings;
	Vector vec(args_->dim);
	dict_->getSubwords(word, ngrams, substrings);
	for (int32_t i = 0; i < ngrams.size(); i++) {
		vec.zero();
		if (ngrams[i] >= 0) {
			vec.addRow(*input_, ngrams[i]);
		}
		std::ostringstream oss;
		oss << vec;
		sub_vec.push_back(substrings[i]);
	}
}

void FastText::ngramVectors(std::string word, std::string& res, std::string& sub) {
	std::vector<int32_t> ngrams;
	std::vector<std::string> substrings;
	Vector vec(args_->dim);
	dict_->getSubwords(word, ngrams, substrings);
	vec.zero();
	int max_pos = 0;
	int max_len = 0;
	for (int32_t i = 0; i < ngrams.size(); i++) {
	    if (ngrams[i] >= 0) {
		  if(substrings[i].size() > max_len) {
			max_len = substrings[i].size();
			max_pos = i;
		  }
	    }
	}
	if(ngrams.size() > 0 && max_pos < ngrams.size() && ngrams[max_pos] >= 0) {
		vec.addRow(*input_, ngrams[max_pos]);
		sub = substrings[max_pos];
		std::ostringstream oss;
		oss << vec;
		res = oss.str();
	}
}
	
void FastText::ngramVectors(std::string word, std::vector<int>& is_word, std::vector<std::string>& sub) {
	std::vector<int32_t> is_word_flags;
	std::vector<std::string> substrings;
	Vector vec(args_->dim);
	dict_->getNgrams_is(word, is_word_flags, substrings);
	is_word = is_word_flags;
	sub = substrings;
}

void FastText::ngramVectors_2(std::string word, std::vector<std::string>& sub) {
	std::vector<std::string> substrings;
	Vector vec(args_->dim);
	dict_->getNgrams_is_2(word, substrings);
	sub = substrings;
}

void FastText::getWordVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);

  double start = 0, end = 0;
  /*start = pub_time_clock();
  for (int i = 0; i < 1000000; ++ i)
  {
	  dict_->getSubwords(word);
  }
  end = pub_time_clock();
  std::cout << "getsubwords time: " << end - start << std::endl;*/
  vec.zero();
  for (int i = 0; i < ngrams.size(); i ++) {
    addInputVector(vec, ngrams[i]);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::getInputVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
  
  if(ngrams.size() == 0) {
    int32_t i = dict_->getId(word);
    vec.zero();
    if(i >= 0) {
      addInputVector(vec, i);
    }
  }
  else {
    vec.zero();
    for (int i = 0; i < ngrams.size(); i ++) {
      addInputVector(vec, ngrams[i]);
    }
    if (ngrams.size() > 0) {
      vec.mul(1.0 / ngrams.size());
    }
  }
}

void FastText::getVector(Vector& vec, const std::string& word) const {
  getWordVector(vec, word);
}

void FastText::getSubwordVector(Vector& vec, const std::string& subword)
    const {
  vec.zero();
  int32_t h = dict_->hash(subword) % args_->bucket;
  h = h + dict_->nwords();
  addInputVector(vec, h);
}

void FastText::saveVectors() {
  std::ofstream ofs(args_->output + ".vec");
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        args_->output + ".vec" + " cannot be opened for saving vectors!");
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    if(word.find("p#") == 0) {
        continue;
    }
    if(word.find("s#") == 0) {
        continue;
    }
    //getWordVector(vec, word);
    getInputVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput() {
  std::ofstream ofs(args_->output + ".output");
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        args_->output + ".output" + " cannot be opened for saving vectors!");
  }
  if (quant_) {
    throw std::invalid_argument(
        "Option -saveOutput is not supported for quantized models.");
  }
  int32_t n = (args_->model == model_name::sup) ? dict_->nlabels()
                                                : dict_->nwords();
  ofs << n << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < n; i++) {
    std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                         : dict_->getWord(i);
	
    if(word.find("p#") == 0) {
        continue;
    }
    if(word.find("s#") == 0) {
        continue;
    }

    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

struct U {
    string pq;
    size_t pid;
    float* cur;
    DATrie* pngramdat;
};

string reverse(string word) {
    string str;
    std::string ngram;
    if(word.size() == 0) return str;
    //if((word[0] & 0xC0) == 0x80) return str;
    for (size_t j = 0; j < word.size();) {
        ngram.push_back(word[j++]);
        while (j < word.size() && (word[j] & 0xC0) == 0x80) {
            ngram.push_back(word[j++]);
        }
        str = ngram + str;
        ngram = "";
    }
    return str;
}

int datrie_foreach_cal(const char* key, size_t value, void* udata) {
    real* pos = ((U*)(udata))->cur;
    //cout << key << " " << value << " " << ((U*)udata)->pq << " " << ((U*)udata)->pid << endl;
    string pq = ((U*)udata)->pq;
    size_t pid = ((U*)udata)->pid;
    //cout << pq << " " << pid << endl;
    DATrie* pngramdat = ((U*)udata)->pngramdat;
    
    real* pcur = pos + pid * 128;
    real pnorm = 0.0;
	for (int j = 0; j < 128; ++ j) {
        //cout << j << " " << pcur[j] << endl;
        pnorm += pcur[j] * pcur[j];
    }
    pnorm = sqrt(pnorm);
    //cout << "pnorm " << pnorm << endl;

    real* cur = (real*)(pos + value * 128);
    real norm = 0.0;
	for (int j = 0; j < 128; ++ j) {
        //cout << j << " " << cur[j] << endl;
        norm += cur[j] * cur[j];
    }
    norm = sqrt(norm);
    //cout << "norm " << norm << endl;
    
    real sim = 0.0;
    if(pnorm > norm)
        sim = norm / pnorm;
    else
        sim = pnorm / norm;
    
    string rkey = reverse(key);
    //cout << sim << endl;
    //cout << rkey << " " << pid << endl;

    if(sim >= 0.99) {
        datrie_insert_key(pngramdat, rkey.c_str(), pid);
        ((U*)udata)->pq = key;
        ((U*)udata)->pid = pid;
        //((U*)udata)->pngramdat = pngramdat;
    } else {
        datrie_insert_key(pngramdat, rkey.c_str(), value);
        ((U*)udata)->pq = key;
        ((U*)udata)->pid = value;
        //((U*)udata)->pngramdat = pngramdat;
    }
    //std::cout << key << "\t" << norm << std::endl;
    return 1;
}

struct ID {
    int32_t id;
    vector<size_t>* ids;
    map<size_t, size_t>* id_map;
    DATrie* pngramdat;
};

int datrie_foreach_id(const char* key, size_t value, void* udata) {
    int32_t id = ((ID*)udata)->id;
    DATrie* pngramdat = ((ID*)udata)->pngramdat;
    if((((ID*)udata)->id_map)->count(value) > 0) {
        datrie_insert_key(pngramdat, key, (*(((ID*)udata)->id_map))[value]);
    } else {
        id += 1;
        (((ID*)udata)->ids)->push_back(value);
        (*(((ID*)udata)->id_map))[value] = id;
        ((ID*)udata)->id = id;
        datrie_insert_key(pngramdat, key, id);
    }
    return 1;
}


void FastText::saveIndex() {
    std::string path(args_->output + ".z");
	std::ofstream ofs(path, std::ofstream::binary);
	if (!ofs.is_open()) {
		throw std::invalid_argument(path + " cannot be opened for saving!");
	}

	char* cur = (char*)(input_->data() + (size_t)dict_->nwords() * args_->dim);

	real* rcur = (real*)(input_->data() + (size_t)dict_->nwords() * args_->dim);
    DATrie* pngramdat = datrie_new();
    DATrie* qngramdat = datrie_new();
    DATrie* ongramdat = datrie_new();
    U s;
    s.cur = rcur;
    s.pid = 0;
    s.pq = "";
    s.pngramdat = pngramdat;

    /*Change from Separate Execution to Overall Execution*/
    //DATrie* ngramdat = datrie_load_from_file("ngramdat.bin");
    DATrie* ngramdat = dict_->ngramdat;
    datrie_foreach_keys(ngramdat, datrie_foreach_cal, &s);

    /*datrie_delete when destroy dict_*/
	//datrie_delete(ngramdat);

    ngramdat = s.pngramdat;
    U t;
    t.cur = rcur;
    t.pid = 0;
    t.pq = "";
    t.pngramdat = qngramdat;
    datrie_foreach_keys(ngramdat, datrie_foreach_cal, &t);
	datrie_delete(ngramdat);
    ngramdat = t.pngramdat;
    

    ID o;
    o.id = -1;
    o.pngramdat = ongramdat;
    o.ids = new vector<size_t>();
    o.id_map = new map<size_t, size_t>();
    datrie_foreach_keys(ngramdat, datrie_foreach_id, &o);


	int32_t tmp = dict_->nwords();
	ofs.write((char*)&tmp, sizeof(tmp));
	tmp = (o.ids)->size();
	ofs.write((char*)&tmp, sizeof(tmp));
	ofs.write((char*)&args_->dim, sizeof(args_->dim));
	ofs.write((char*)&args_->minn, sizeof(args_->minn));
	ofs.write((char*)&args_->maxn, sizeof(args_->maxn)); 
	DATrie* worddat = datrie_new();
  	Vector vec(args_->dim);
	for (int32_t i = 0; i < dict_->nwords(); i++) {
		std::string word = dict_->getWord(i);
		//getWordVector(vec, word);
		getInputVector(vec, word);
		datrie_insert_key(worddat, word.c_str(), i);
		ofs.write((char*)vec.data_, vec.size() * sizeof(*vec.data_));
	}

    cout << (o.ids)->size() << endl;
    for(int32_t i = 0; i < (o.ids)->size(); i++) {
        Vector m_vec(args_->dim);
        for(int32_t j = 0; j < 128; j++) {
            m_vec[j] = (rcur + (*(o.ids))[i] * args_->dim)[j];
        }
	    ofs.write((char*)m_vec.data_, m_vec.size() * sizeof(*m_vec.data_));
    }
	datrie_delete(ngramdat);
    ngramdat = o.pngramdat;


	DASTrie* dasworddat = dastrie_new();
	dastrie_build_convert(dasworddat, worddat);

	size_t size = 0;
	void* buffer = dastrie_map_to_buffer(dasworddat, &size);
	cur = (char*)buffer;
    std::streamsize ws = size;
    std::streamsize batch = 512000000;
	for (int32_t i = 0; i < ws / batch; ++ i)
	{
		ofs.write(cur, batch);
		cur += batch;
	}
	if (ws % batch) ofs.write(cur, ws % batch);
    std::cout << "nwords_size: " << (int32_t)dastrie_get_key_num(dasworddat) << std::endl;
	datrie_delete(worddat);
	dastrie_delete(dasworddat);

	DASTrie* dasngramdat = dastrie_new();
	dastrie_build_convert(dasngramdat, ngramdat);
	size = 0;
	buffer = dastrie_map_to_buffer(dasngramdat, &size);
	cur = (char*)buffer;
	ws = size;
	for (int32_t i = 0; i < ws / batch; ++ i)
	{
		ofs.write(cur, batch);
		cur += batch;
	}
	if (ws % batch) ofs.write(cur, ws % batch);
    std::cout << "nngrams_size: " << (int32_t)dastrie_get_key_num(dasngramdat) << std::endl;
	datrie_delete(ngramdat);
	dastrie_delete(dasngramdat);
	pub_free(buffer);
    delete o.ids;
    delete o.id_map;

	ofs.close();
}


void FastText::saveIndex(std::string path) {
    //std::string path(args_->output + ".z");
	std::ofstream ofs(path, std::ofstream::binary);
	if (!ofs.is_open()) {
		throw std::invalid_argument(path + " cannot be opened for saving!");
	}

	char* cur = (char*)(input_->data() + (size_t)dict_->nwords() * args_->dim);

	real* rcur = (real*)(input_->data() + (size_t)dict_->nwords() * args_->dim);
    DATrie* pngramdat = datrie_new();
    DATrie* qngramdat = datrie_new();
    DATrie* ongramdat = datrie_new();
    U s;
    s.cur = rcur;
    s.pid = 0;
    s.pq = "";
    s.pngramdat = pngramdat;

    /*Change from Separate Execution to Overall Execution*/
    DATrie* ngramdat = dict_->ngramdat;
    //DATrie* ngramdat = datrie_load_from_file("cut_ngramdat.bin");
    //DATrie* ngramdat = dict_->ngramdat;
    datrie_foreach_keys(ngramdat, datrie_foreach_cal, &s);

    /*datrie_delete when destroy dict_*/
	//datrie_delete(ngramdat);

    ngramdat = s.pngramdat;
    U t;
    t.cur = rcur;
    t.pid = 0;
    t.pq = "";
    t.pngramdat = qngramdat;
    datrie_foreach_keys(ngramdat, datrie_foreach_cal, &t);
	datrie_delete(ngramdat);
    ngramdat = t.pngramdat;
    

    ID o;
    o.id = -1;
    o.pngramdat = ongramdat;
    o.ids = new vector<size_t>();
    o.id_map = new map<size_t, size_t>();
    datrie_foreach_keys(ngramdat, datrie_foreach_id, &o);


	int32_t tmp = dict_->nwords();
	ofs.write((char*)&tmp, sizeof(tmp));
	tmp = (o.ids)->size();
	ofs.write((char*)&tmp, sizeof(tmp));
	ofs.write((char*)&args_->dim, sizeof(args_->dim));
	ofs.write((char*)&args_->minn, sizeof(args_->minn));
	ofs.write((char*)&args_->maxn, sizeof(args_->maxn)); 
	DATrie* worddat = datrie_new();
  	Vector vec(args_->dim);
	for (int32_t i = 0; i < dict_->nwords(); i++) {
		std::string word = dict_->getWord(i);
		getInputVector(vec, word);
		datrie_insert_key(worddat, word.c_str(), i);
		ofs.write((char*)vec.data_, vec.size() * sizeof(*vec.data_));
	}

    cout << (o.ids)->size() << endl;
    for(int32_t i = 0; i < (o.ids)->size(); i++) {
        Vector m_vec(args_->dim);
        for(int32_t j = 0; j < 128; j++) {
            m_vec[j] = (rcur + (*(o.ids))[i] * args_->dim)[j];
        }
	    ofs.write((char*)m_vec.data_, m_vec.size() * sizeof(*m_vec.data_));
    }
	datrie_delete(ngramdat);
    ngramdat = o.pngramdat;


	DASTrie* dasworddat = dastrie_new();
	dastrie_build_convert(dasworddat, worddat);

	size_t size = 0;
	void* buffer = dastrie_map_to_buffer(dasworddat, &size);
	cur = (char*)buffer;
    std::streamsize ws = size;
    std::streamsize batch = 512000000;
	for (int32_t i = 0; i < ws / batch; ++ i)
	{
		ofs.write(cur, batch);
		cur += batch;
	}
	if (ws % batch) ofs.write(cur, ws % batch);
    std::cout << "nwords_size: " << (int32_t)dastrie_get_key_num(dasworddat) << std::endl;
	datrie_delete(worddat);
	dastrie_delete(dasworddat);
    
	DASTrie* dasngramdat = dastrie_new();
    std::cout<< "2" << std::endl;
	dastrie_build_convert(dasngramdat, ngramdat);
    std::cout<< "3" << std::endl;

	size = 0;
    //std::cout<< "4" << std::endl;
	buffer = dastrie_map_to_buffer(dasngramdat, &size);
	//buffer = datrie_map_to_buffer(ngramdat, &size);
    //std::cout<< "5" << std::endl;
	cur = (char*)buffer;
	ws = size;
    //std::cout<< "6" << std::endl;
	for (int32_t i = 0; i < ws / batch; ++ i)
	{
		ofs.write(cur, batch);
		cur += batch;
	}
    //std::cout<< "7" << std::endl;
	if (ws % batch) ofs.write(cur, ws % batch);
    //std::cout<< "8" << std::endl;
    std::cout << "nngrams_size: " << (int32_t)dastrie_get_key_num(dasngramdat) << std::endl;
    std::cout << "nngrams_size: " << (int32_t)datrie_get_key_num(ngramdat) << std::endl;
	datrie_delete(ngramdat);
	dastrie_delete(dasngramdat);
	pub_free(buffer);
    delete o.ids;
    delete o.id_map;

	ofs.close();
}

bool FastText::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void FastText::signModel(std::ostream& out) {
  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void FastText::saveModel() {
  std::string fn(args_->output);
  if (quant_) {
    fn += ".ftz";
  } else {
    fn += ".bin";
  }
  saveModel(fn);
}

void FastText::saveModel(const std::string path) {
  std::ofstream ofs(path, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::invalid_argument(path + " cannot be opened for saving!");
  }
  signModel(ofs);
  args_->save(ofs);
  dict_->save(ofs);

  ofs.write((char*)&(quant_), sizeof(bool));
  if (quant_) {
    qinput_->save(ofs);
  } else {
    input_->save(ofs);
  }

  ofs.write((char*)&(args_->qout), sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->save(ofs);
  } else {
    output_->save(ofs);
  }

  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  qinput_ = std::make_shared<QMatrix>();
  qoutput_ = std::make_shared<QMatrix>();
  args_->load(in);
  if (version == 11 && args_->model == model_name::sup) {
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
  }
  dict_->load(in);

  bool quant_input;
  in.read((char*) &quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    qinput_->load(in);
  } else {
    input_->load(in);
  }

  if (!quant_input && dict_->isPruned()) {
    throw std::invalid_argument(
        "Invalid model file.\n"
        "Please download the updated model from www.fasttext.cc.\n"
        "See issue #332 on Github for more information.\n");
  }

  in.read((char*) &args_->qout, sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->load(in);
  } else {
    output_->load(in);
  }

  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);

  //if (args_->model == model_name::sup) {
  //  model_->setTargetCounts(dict_->getCounts(entry_type::label));
  //} else {
  //  model_->setTargetCounts(dict_->getCounts(entry_type::word));
  //}
}

void FastText::printInfo(real progress, real loss) {
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst_real = real(tokenCount_) / t;
  int32_t wst = wst_real;
  real lr = args_->lr * (1.0 - progress);
  int32_t eta;
  if (progress == 0 || t == 0) {
    eta = 720 * 3600; // One month.
  } else {
    eta = int32_t((t / progress) * (1 - progress) / args_->thread);
  }
  int32_t etah = eta / 3600;
  int32_t etam = (eta - etah * 3600) / 60;
  progress = progress * 100;
  std::cerr << std::fixed;
  std::cerr << "\rProgress: ";
  std::cerr << std::setprecision(1) << std::setw(6) << progress << "%";
  std::cerr << " words/s: " << std::setw(9) << wst;
  std::cerr << " lr: " << std::setw(9) << std::setprecision(6) << lr;
  std::cerr << " loss: " << std::setw(9) << std::setprecision(6) << loss;
  std::cerr << " eta: " << std::setw(3) << etah;
  std::cerr << "h" << std::setw(2) << etam << "m";
  std::cerr << std::flush;
}

std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const {
  Vector norms(input_->m_);
  input_->l2NormRow(norms);
  std::vector<int32_t> idx(input_->m_, 0);
  std::iota(idx.begin(), idx.end(), 0);
  auto eosid = dict_->getId(Dictionary::EOS);
  std::sort(idx.begin(), idx.end(),
      [&norms, eosid] (size_t i1, size_t i2) {
      return eosid ==i1 || (eosid != i2 && norms[i1] > norms[i2]);
      });
  idx.erase(idx.begin() + cutoff, idx.end());
  return idx;
}

void FastText::quantize(std::shared_ptr<Args> qargs) {
  if (args_->model != model_name::sup) {
    throw std::invalid_argument(
        "For now we only support quantization of supervised models");
  }
  args_->input = qargs->input;
  args_->qout = qargs->qout;
  args_->output = qargs->output;

  if (qargs->cutoff > 0 && qargs->cutoff < input_->m_) {
    auto idx = selectEmbeddings(qargs->cutoff);
    dict_->prune(idx);
    std::shared_ptr<Matrix> ninput =
        std::make_shared<Matrix>(idx.size(), args_->dim);
    for (auto i = 0; i < idx.size(); i++) {
      for (auto j = 0; j < args_->dim; j++) {
        ninput->at(i, j) = input_->at(idx[i], j);
      }
    }
    input_ = ninput;
    if (qargs->retrain) {
      args_->epoch = qargs->epoch;
      args_->lr = qargs->lr;
      args_->thread = qargs->thread;
      args_->verbose = qargs->verbose;
      startThreads();
    }
  }

  qinput_ = std::make_shared<QMatrix>(*input_, qargs->dsub, qargs->qnorm);

  if (args_->qout) {
    qoutput_ = std::make_shared<QMatrix>(*output_, 2, qargs->qnorm);
  }

  quant_ = true;
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);
}

void FastText::supervised(
    Model& model,
    real lr,
    const std::vector<int32_t>& line,
    const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

void FastText::cbow(Model& model, real lr,
                    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 1; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    //ngrams.push_back(line[0]);
    std::vector<int32_t> ngrams_with_pos(ngrams);
    ngrams_with_pos.push_back(line[0]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 1 && w + c < line.size()) {
        model.update(ngrams_with_pos, line[w + c], lr);
        //model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line, const std::vector<int32_t>& src) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 1; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    //ngrams.push_back(line[0]);
    std::vector<int32_t> ngrams_with_pos_src(ngrams);
    ngrams_with_pos_src.push_back(line[0]);
    if(src[w] >= 0) {
    	ngrams_with_pos_src.push_back(src[w]);
    }
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 1 && w + c < line.size()) {
        model.update(ngrams_with_pos_src, line[w + c], lr);
        //model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::test(std::istream& in, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF) {
    dict_->getLine(in, line, labels, model_->rng);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::cout << "N" << "\t" << nexamples << std::endl;
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << "\t" << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << "\t" << precision / nlabels << std::endl;
  std::cerr << "Number of examples: " << nexamples << std::endl;
}

void FastText::predict(std::istream& in, int32_t k,
                       std::vector<std::pair<real,std::string>>& predictions) const {
  std::vector<int32_t> words, labels;
  predictions.clear();
  dict_->getLine(in, words, labels, model_->rng);
  predictions.clear();
  if (words.empty()) return;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());
  std::vector<std::pair<real,int32_t>> modelPredictions;
  model_->predict(words, k, modelPredictions, hidden, output);
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
    predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
  std::vector<std::pair<real,std::string>> predictions;
  while (in.peek() != EOF) {
    predictions.clear();
    predict(in, k, predictions);
    if (predictions.empty()) {
      std::cout << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << " ";
      }
      std::cout << it->second;
      if (print_prob) {
        std::cout << " " << std::exp(it->first);
      }
    }
    std::cout << std::endl;
  }
}

void FastText::getSentenceVector(
    std::istream& in,
    fasttext::Vector& svec) {
  svec.zero();
  if (args_->model == model_name::sup) {
    std::vector<int32_t> line, labels;
    dict_->getLine(in, line, labels, model_->rng);
    for (int32_t i = 0; i < line.size(); i++) {
      addInputVector(svec, line[i]);
    }
    if (!line.empty()) {
      svec.mul(1.0 / line.size());
    }
  } else {
    Vector vec(args_->dim);
    std::string sentence;
    std::getline(in, sentence);
    std::istringstream iss(sentence);
    std::string word;
    int32_t count = 0;
    while (iss >> word) {
      getWordVector(vec, word);
      real norm = vec.norm();
      if (norm > 0) {
        vec.mul(1.0 / norm);
        svec.addVector(vec);
        count++;
      }
    }
    if (count > 0) {
      svec.mul(1.0 / count);
    }
  }
}

void FastText::ngramVectors(std::string word) {
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  Vector vec(args_->dim);
  dict_->getSubwords(word, ngrams, substrings);
  for (int32_t i = 0; i < ngrams.size(); i++) {
    vec.zero();
    if (ngrams[i] >= 0) {
      if (quant_) {
        vec.addRow(*qinput_, ngrams[i]);
      } else {
        vec.addRow(*input_, ngrams[i]);
      }
    }
    std::cout << ngrams[i] << " " << substrings[i] << " " << vec << std::endl;
  }
}

void FastText::precomputeWordVectors(Matrix& wordVectors) {
  Vector vec(args_->dim);
  wordVectors.zero();
  std::cerr << "Pre-computing word vectors...";
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      wordVectors.addRow(vec, i, 1.0 / norm);
    }
  }
  std::cerr << " done." << std::endl;
}

void FastText::findNN(const Matrix& wordVectors, const Vector& queryVec,
                      int32_t k, const std::set<std::string>& banSet) {
  real queryNorm = queryVec.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }
  std::priority_queue<std::pair<real, std::string>> heap;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    real dp = wordVectors.dotRow(queryVec, i);
    heap.push(std::make_pair(dp / queryNorm, word));
  }
  int32_t i = 0;
  while (i < k && heap.size() > 0) {
    auto it = banSet.find(heap.top().second);
    if (it == banSet.end()) {
      std::cout << heap.top().second << " " << heap.top().first << std::endl;
      i++;
    }
    heap.pop();
  }
}

void FastText::nn(int32_t k) {
  std::string queryWord;
  Vector queryVec(args_->dim);
  Matrix wordVectors(dict_->nwords(), args_->dim);
  precomputeWordVectors(wordVectors);
  std::set<std::string> banSet;
  std::cout << "Query word? ";
  while (std::cin >> queryWord) {
    banSet.clear();
    banSet.insert(queryWord);
    getWordVector(queryVec, queryWord);
    findNN(wordVectors, queryVec, k, banSet);
    std::cout << "Query word? ";
  }
}

void FastText::analogies(int32_t k) {
  std::string word;
  Vector buffer(args_->dim), query(args_->dim);
  Matrix wordVectors(dict_->nwords(), args_->dim);
  precomputeWordVectors(wordVectors);
  std::set<std::string> banSet;
  std::cout << "Query triplet (A - B + C)? ";
  while (true) {
    banSet.clear();
    query.zero();
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, -1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);

    findNN(wordVectors, query, k, banSet);
    std::cout << "Query triplet (A - B + C)? ";
  }
}

void FastText::trainThread(int32_t threadId) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, threadId);
  if (args_->model == model_name::sup) {
    model.setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model.setTargetCounts(dict_->getCounts(entry_type::word));
  }

  const int64_t ntokens = dict_->ntokens();
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels, src;
  while (tokenCount_ < args_->epoch * ntokens) {
    real progress = real(tokenCount_) / (args_->epoch * ntokens);
    real lr = args_->lr * (1.0 - progress);
    if (args_->model == model_name::sup) {
      localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
      supervised(model, lr, line, labels);
    } else if (args_->model == model_name::cbow) {
      localTokenCount += dict_->getLine(ifs, line, model.rng);
      cbow(model, lr, line);
    } else if (args_->model == model_name::sg) {
      localTokenCount += dict_->getLine_pos_src(ifs, line, src, model.rng);
      skipgram(model, lr, line, src);
    }
    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount_ += localTokenCount;
      localTokenCount = 0;
      if (threadId == 0 && args_->verbose > 1) {
        printInfo(progress, model.getLoss());
      }
    }
  }
  if (threadId == 0 && args_->verbose > 0) {
    printInfo(1.0, model.getLoss());
    std::cerr << std::endl;
  }
  ifs.close();
}

void FastText::loadVectors(std::string filename) {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    throw std::invalid_argument(
        "Dimension of pretrained vectors (" + std::to_string(dim) +
        ") does not match dimension (" + std::to_string(args_->dim) + ")!");
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->data_[i * dim + j];
    }
  }
  in.close();

  dict_->threshold(1, 0);
  input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords()) continue;
    for (size_t j = 0; j < dim; j++) {
      input_->data_[idx * dim + j] = mat->data_[i * dim + j];
    }
  }
}

void FastText::train(std::shared_ptr<Args> args) {
  args_ = args;
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    throw std::invalid_argument("Cannot use stdin for training!");
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    throw std::invalid_argument(
        args_->input + " cannot be opened for training!");
  }
  dict_->readFromFile(ifs);
  ifs.close();

  if (args_->pretrainedVectors.size() != 0) {
    loadVectors(args_->pretrainedVectors);
  } else {
    /*input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);*/
    input_ = std::make_shared<Matrix>(dict_->nwords()+dict_->nngrams(), args_->dim);

    input_->uniform(1.0 / args_->dim);
  }

  if (args_->model == model_name::sup) {
    output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
  } else {
    output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
  }
  output_->zero();
  startThreads();
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::startThreads() {
  start = clock();
  tokenCount_ = 0;
  if (args_->thread > 1) {
    std::vector<std::thread> threads;
    for (int32_t i = 0; i < args_->thread; i++) {
      threads.push_back(std::thread([=]() { trainThread(i); }));
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
      it->join();
    }
  } else {
    trainThread(0);
  }
}

int FastText::getDimension() const {
    return args_->dim;
}

bool FastText::isQuant() const {
  return quant_;
}

}
