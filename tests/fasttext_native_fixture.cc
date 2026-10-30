// Deterministic correctness fixture; not compression or model-quality evidence.
#include "fasttext.h"
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace fasttext;
class FixtureDictionary : public Dictionary {
 public:
  explicit FixtureDictionary(std::shared_ptr<Args> args):Dictionary(args) {
    worddat=datrie_new(); ngramdat=datrie_new();
    for (const std::string& word:{"alpha","alphabet","beta","你好","</s>","__label__left","__label__right"}) add(word);
    threshold(1,1); initTableDiscard(); initNgrams();
  }
};
class Fixture : public FastText {
 public:
  explicit Fixture(int dim, bool hs=false) {
    args_=std::make_shared<Args>(); args_->dim=dim; args_->minn=2; args_->maxn=4;
    args_->model=model_name::sup; args_->loss=hs ? loss_name::hs : loss_name::softmax;
    dict_=std::make_shared<FixtureDictionary>(args_);
    input_=std::make_shared<Matrix>(dict_->nwords()+dict_->nngrams(),dim);
    output_=std::make_shared<Matrix>(dict_->nlabels(),dim);
    for (int64_t i=0;i<input_->m_;++i)
      for(int j=0;j<dim;++j) input_->at(i,j)=float((i*17+j*31)%113-56)/13.f;
    for (int64_t i=0;i<output_->m_;++i)
      for(int j=0;j<dim;++j) output_->at(i,j)=float((i*7+j*3)%17-8)/19.f;
    model_=std::make_shared<Model>(input_,output_,args_,0); model_->quant_=false;
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
    for (int i=0;i<5;++i) model_->update(dict_->getSubwords("alpha"),i%2,.01f);
  }
};
void same(const Vector& a,const Vector& b) {
  assert(a.size()==b.size()); assert(std::memcmp(a.data_,b.data_,a.size()*4)==0);
}
int main(int argc,char** argv) {
  assert(argc>=3);
  if(std::string(argv[1])=="create") {
    Fixture model(std::stoi(argv[3]),argc>4 && std::string(argv[4])=="hs"); model.saveModel(argv[2]); return 0;
  }
  assert(argc==4);
  FastText dense,native; dense.loadModel(argv[2]); native.loadModel(argv[3]);
  assert(!dense.hasNativeInput() && native.hasNativeInput());
  int dim=dense.getDimension(); Vector a(dim),b(dim);
  auto matrix=dense.getInputMatrix();
  for(int64_t id=0;id<matrix->m_;++id) {
    dense.getInputVector(a,id); native.getInputVector(b,id); same(a,b);
  }
  for(const std::string& word:{"alpha","alphabet","beta","你好","alphax","","☃unknown","</s>"}) {
    assert(dense.getWordId(word)==native.getWordId(word));
    assert(dense.getDictionary()->getSubwords(word)==native.getDictionary()->getSubwords(word));
    dense.getWordVector(a,word); native.getWordVector(b,word); same(a,b);
    dense.getInputVector(a,word); native.getInputVector(b,word); same(a,b);
  }
  for(const std::string& key:{"al","pha","<a","你好","missing☃"}) {
    assert(dense.getSubwordId(key)==dense.getDictionary()->getNgramId(key));
    assert(dense.getSubwordId(key)==native.getSubwordId(key));
    dense.getSubwordVector(a,key); native.getSubwordVector(b,key); same(a,b);
  }
  for(const std::string& sentence:{"alpha beta alpha","你好 alpha","","☃unknown"}) {
    std::istringstream left(sentence),right(sentence);
    dense.getSentenceVector(left,a); native.getSentenceVector(right,b); same(a,b);
  }
  std::vector<int32_t> ids;
  for(int i=0;i<517;++i) ids.push_back((i*7)%matrix->m_);
  a.zero(); b.zero(); dense.addInputVectors(a,ids); native.addInputVectors(b,ids); same(a,b);
  std::istringstream left("alpha beta"),right("alpha beta");
  std::vector<std::pair<real,std::string>> p,q;
  dense.predict(left,2,p); native.predict(right,2,q); assert(p==q && !p.empty());
  bool rejected=false;
  try {native.getInputMatrix();} catch(const std::exception&) {rejected=true;}
  assert(rejected);
  rejected=false;
  try {native.saveModel(std::string(argv[3])+"/forbidden.bin");} catch(const std::exception&) {rejected=true;}
  assert(rejected);
  rejected=false;
  try {native.loadNativeBundle(std::string(argv[3])+"/missing");} catch(const std::exception&) {rejected=true;}
  assert(rejected && native.hasNativeInput());
  native.getWordVector(b,"alpha"); dense.getWordVector(a,"alpha"); same(a,b);
  native.loadModel(argv[2]); assert(!native.hasNativeInput());
  native.getWordVector(b,"alpha"); same(a,b);
  std::cout<<"All original rows, IDs, OOV/UTF-8, ordered/chunked aggregation, sentence and label probabilities exact.\n";
}
