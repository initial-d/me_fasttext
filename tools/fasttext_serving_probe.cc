// CPU evidence driver for actual me_fasttext training and end-to-end serving.
// Query TSV: w<TAB>word or s<TAB>sentence. Output oracle is raw float32 bytes.
#include "fasttext.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
using namespace fasttext;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
void memory(std::ostream& out) {
  std::ifstream in("/proc/self/status"); std::string line;
  out<<"{"; bool first=true;
  while(std::getline(in,line)) {
    auto colon=line.find(':'); std::string key=line.substr(0,colon);
    if(key!="VmRSS" && key!="RssAnon" && key!="RssFile") continue;
    if(!first) out<<","; first=false;
    std::istringstream value(line.substr(colon+1)); long kb; value>>kb;
    out<<"\""<<key<<"_bytes\":"<<kb*1024;
  }
  out<<"}";
}
void distribution(std::ostream& out,std::vector<double> values) {
  std::sort(values.begin(),values.end());
  if(values.empty()) throw std::runtime_error("empty timing distribution");
  out<<"{\"count\":"<<values.size()<<",\"p50_us\":"<<values[(values.size()-1)/2]
     <<",\"p95_us\":"<<values[(values.size()-1)*95/100]<<"}";
}
int main(int argc,char** argv) {
  if(argc<2) throw std::runtime_error("train corpus model report | serve model queries oracle report");
  if(std::string(argv[1])=="train") {
    if(argc!=5) throw std::runtime_error("train corpus model report");
    auto args=std::make_shared<Args>(); args->input=argv[2]; args->output=std::string(argv[3])+".unused";
    args->model=model_name::sg; args->loss=loss_name::ns; args->dim=64;
    args->minn=2; args->maxn=12; args->minCount=2; args->ws=3; args->neg=5;
    args->epoch=2; args->thread=1; args->t=1; args->lr=.025; args->verbose=0;
    FastText model; auto start=Clock::now(); model.train(args); double elapsed=seconds(start);
    auto input=model.getInputMatrix(); Matrix initial(input->m_,input->n_); initial.uniform(1.0/args->dim);
    int64_t changed=0,changedNgrams=0;
    for(int64_t i=0;i<input->m_;++i) {
      bool different=std::memcmp(input->data()+i*input->n_,initial.data()+i*input->n_,input->n_*4)!=0;
      changed+=different; changedNgrams+=different && i>=model.getDictionary()->nwords();
    }
    model.saveModel(argv[3]); std::ofstream out(argv[4]); out<<std::setprecision(10);
    out<<"{\"training_seconds\":"<<elapsed<<",\"rows\":"<<input->m_<<",\"dim\":"<<input->n_
       <<",\"word_rows\":"<<model.getDictionary()->nwords()<<",\"ngram_rows\":"<<model.getDictionary()->nngrams()
       <<",\"changed_rows\":"<<changed<<",\"changed_ngram_rows\":"<<changedNgrams
       <<",\"epochs\":2,\"threads\":1,\"min_count\":2,\"minn\":2,\"maxn\":12,\"window\":3,\"negative_samples\":5,\"lr\":0.025,\"initializer_seed\":1,\"training_rng_seed\":0}\n";
    return 0;
  }
  if(std::string(argv[1])!="serve" || argc!=6) throw std::runtime_error("serve model queries oracle report");
  std::ifstream queries(argv[3]); std::vector<std::pair<char,std::string>> trace; std::string line;
  while(std::getline(queries,line)) {
    if(line.size()<3 || line[1]!='\t' || (line[0]!='w' && line[0]!='s')) throw std::runtime_error("invalid query");
    trace.emplace_back(line[0],line.substr(2));
  }
  if(trace.empty()) throw std::runtime_error("empty query trace");
  auto start=Clock::now(); FastText model; model.loadModel(argv[2]); double load=seconds(start);
  std::ofstream report(argv[5]); report<<std::setprecision(10);
  report<<"{\"native\":"<<(model.hasNativeInput()?"true":"false")<<",\"load_seconds\":"<<load<<",\"after_load\":";
  memory(report); report<<",\"passes\":[";
  Vector vector(model.getDimension()); std::vector<real> oracle;
  // Keep allocation and oracle file I/O outside the timed query calls.
  oracle.reserve(trace.size()*model.getDimension());
  for(int pass=0;pass<4;++pass) {
    std::vector<double> word,sentence;
    word.reserve(trace.size()); sentence.reserve(trace.size());
    auto batch=Clock::now();
    for(const auto& query:trace) {
      std::istringstream text(query.second); auto begin=Clock::now();
      if(query.first=='w') model.getWordVector(vector,query.second);
      else model.getSentenceVector(text,vector);
      double us=seconds(begin)*1e6; (query.first=='w'?word:sentence).push_back(us);
      if(pass==0) oracle.insert(oracle.end(),vector.data_,vector.data_+vector.size());
    }
    double batchTime=seconds(batch);
    if(pass) report<<",";
    report<<"{\"batch_seconds\":"<<batchTime<<",\"word\":"; distribution(report,word);
    report<<",\"sentence\":"; distribution(report,sentence); report<<",\"memory\":"; memory(report); report<<"}";
  }
  struct rusage usage; getrusage(RUSAGE_SELF,&usage);
  report<<"],\"peak_rss_bytes\":"<<usage.ru_maxrss*1024<<",\"oracle_vectors\":"<<trace.size()<<"}\n";
  std::ofstream output(argv[4],std::ios::binary); output.write(reinterpret_cast<char*>(oracle.data()),oracle.size()*4);
  if(!output || !report) throw std::runtime_error("cannot write probe output");
}
