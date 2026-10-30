/* Optional Linux source build: legacy FastText facade over the native row ABI.
 * Metadata/DASTrie files are trusted, self-produced, host-specific artifacts.
 * Native serving is immutable and single-worker, including const query calls. */
#include "fasttext.h"
#include "meft_rows.h"
#include "trie/dastrie.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <limits>
#include <sys/stat.h>

namespace fasttext {
namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void littleHost() {
  uint16_t one = 1;
  require(*reinterpret_cast<unsigned char*>(&one) == 1 && sizeof(real) == 4 &&
          std::numeric_limits<real>::is_iec559, "native FastText requires little-endian IEEE float32");
}
void write64(std::ostream& out, uint64_t n) {
  unsigned char b[8];
  for (size_t i=0; i<8; ++i) b[i] = static_cast<unsigned char>(n >> (8*i));
  out.write(reinterpret_cast<char*>(b), 8);
}
uint64_t read64(std::istream& in) {
  unsigned char b[8];
  in.read(reinterpret_cast<char*>(b), 8);
  require(bool(in), "truncated native metadata");
  uint64_t n=0;
  for (size_t i=0; i<8; ++i) n |= uint64_t(b[i]) << (8*i);
  return n;
}
void magic(std::istream& in, const char* expected) {
  char b[8]; in.read(b,8);
  require(bool(in) && std::memcmp(b,expected,8)==0, "invalid native FastText metadata version");
}
struct StoreDelete { void operator()(meft_rows* p) const { meft_rows_close(p); } };
struct IndexDelete { void operator()(DASTrie* p) const { if (p) dastrie_delete_unsafe(p); } };
class NativeInput : public RowBackend {
  std::unique_ptr<meft_rows,StoreDelete> store_;
  std::unique_ptr<DASTrie,IndexDelete> index_;
  uint64_t count_, dim_, chunk_;
 public:
  NativeInput(const std::string& path, uint64_t rows, uint64_t dim) : count_(rows), dim_(dim) {
    std::ifstream cfg(path+"/reader.cfg",std::ios::binary);
    magic(cfg,"MEFTCFG1");
    uint64_t cache=read64(cfg); chunk_=read64(cfg);
    uint64_t policy=read64(cfg), mode=read64(cfg);
    require(cache<=SIZE_MAX && chunk_>0 && chunk_<=SIZE_MAX/sizeof(real)/dim &&
            policy<=1 && mode<=2 && cfg.peek()==EOF, "invalid native reader settings");
    char error[512];
    store_.reset(meft_rows_open((path+"/table.men").c_str(),error,sizeof(error)));
    require(bool(store_),error);
    require(meft_rows_count(store_.get())==rows && meft_rows_width(store_.get())==dim*4,
            "native table shape differs from model metadata");
    require(meft_rows_set_cache_bytes(store_.get(),static_cast<size_t>(cache))==0 &&
            meft_rows_set_cache_policy(store_.get(),static_cast<int>(policy))==0,"cannot configure native reader");
    index_.reset(dastrie_load_from_file((path+"/keys.das").c_str()));
    require(bool(index_),"cannot load trusted n-gram DASTrie");
  }
  int64_t rows() const override { return static_cast<int64_t>(count_); }
  int64_t width() const override { return static_cast<int64_t>(dim_); }
  int32_t lookup(const std::string& key, int32_t words, int32_t ngrams) const {
    size_t id;
    if (dastrie_search_exact_match(index_.get(),key.c_str(),&id)==-1) return -1;
    require(id>=static_cast<size_t>(words) && id<static_cast<size_t>(words)+ngrams && id<count_,
            "n-gram index ID differs from model registry");
    return static_cast<int32_t>(id);
  }
  size_t keyCount() const { return dastrie_get_key_num(index_.get()); }
  void addRows(Vector& out, const std::vector<int32_t>& ids) const override {
    require(out.size()==width(),"native vector dimension mismatch");
    for (int32_t id:ids) require(id>=0 && uint64_t(id)<count_,"native row ID out of bounds");
    size_t step=static_cast<size_t>(std::min<uint64_t>(chunk_,ids.size()));
    std::vector<uint64_t> query(step);
    std::vector<float> decoded(step*dim_);
    char error[512];
    for (size_t begin=0; begin<ids.size();) {
      size_t n=std::min(step,ids.size()-begin);
      for (size_t i=0;i<n;++i) query[i]=static_cast<uint64_t>(ids[begin+i]);
      require(meft_rows_gather(store_.get(),query.data(),n,decoded.data(),n*dim_*4,error,sizeof(error))==0,error);
      // Same float32 adds, multiplicity and ID order as Vector::addRow. Do not
      // use the FP64 native bag reducer or reassociate partial chunk sums.
      for (size_t i=0;i<n;++i)
        for (size_t j=0;j<dim_;++j) out.data_[j] += decoded[i*dim_+j];
      begin+=n;
    }
  }
};
struct KeyDump {
  std::ostream& out; uint64_t words, rows, written; bool valid;
  KeyDump(std::ostream& stream,uint64_t w,uint64_t r):out(stream),words(w),rows(r),written(0),valid(true) {}
};
int dumpKey(const char* key, size_t id, void* ptr) {
  KeyDump& d=*static_cast<KeyDump*>(ptr);
  if (id>=d.rows-d.words) { d.valid=false; return 0; }
  write64(d.out,d.words+id); write64(d.out,std::strlen(key));
  d.out.write(key,std::strlen(key)); ++d.written;
  // DATrie's callback convention: 1 continues, 0 stops traversal.
  return 1;
}
}

void FastText::dumpNative(const std::string& directory) const {
  littleHost();
  require(!quant_ && !input_backend_ && input_ && output_,"native export requires an unquantized dense me_fasttext model");
  require(!dict_->isPruned() && input_->m_==int64_t(dict_->nwords())+dict_->nngrams() &&
          input_->n_==args_->dim && args_->dim>0,"unsupported pruned or inconsistent input registry");
  require(mkdir(directory.c_str(),0700)==0,"use a new extraction directory");
  std::ofstream raw(directory+"/input.f32",std::ios::binary);
  const char* bytes=reinterpret_cast<const char*>(input_->data());
  uint64_t total=uint64_t(input_->m_)*input_->n_*4;
  for (uint64_t begin=0;begin<total;) {
    size_t n=static_cast<size_t>(std::min<uint64_t>(1048576,total-begin));
    raw.write(bytes+begin,n); begin+=n;
  }
  raw.close(); require(bool(raw),"cannot write native source rows");
  std::ofstream keys(directory+"/keys.bin",std::ios::binary);
  keys.write("MEFTKEY1",8); write64(keys,dict_->nngrams());
  KeyDump dump(keys,uint64_t(dict_->nwords()),uint64_t(input_->m_));
  datrie_foreach_keys(dict_->ngramdat,dumpKey,&dump);
  keys.close();
  require(bool(keys) && dump.valid && dump.written==uint64_t(dict_->nngrams()),"invalid n-gram registry export");
  std::ofstream meta(directory+"/model.meta",std::ios::binary);
  meta.write("MEFTFT01",8); write64(meta,input_->m_); write64(meta,args_->dim);
  args_->save(meta); dict_->save(meta,false); output_->save(meta);
  meta.close(); require(bool(meta),"cannot write native model metadata");
}

void FastText::loadNativeBundle(const std::string& directory) {
  littleHost();
  // Build independent state first; failed opening preserves the existing model.
  std::ifstream meta(directory+"/model.meta",std::ios::binary);
  magic(meta,"MEFTFT01");
  uint64_t rows=read64(meta), dim=read64(meta);
  require(rows>0 && rows<=INT32_MAX && dim>0 && dim<=INT32_MAX,"invalid native model dimensions");
  auto args=std::make_shared<Args>(); args->load(meta);
  require(bool(meta) && args->dim==int64_t(dim) && args->minn>=0 && args->maxn>=args->minn &&
          args->maxn<=INT32_MAX && args->bucket>=0,"invalid native model arguments");
  auto dict=std::make_shared<Dictionary>(args); dict->load(meta,false);
  require(!dict->isPruned() && uint64_t(dict->nwords())+dict->nngrams()==rows,"native dictionary/input row mismatch");
  int64_t m,n; meta.read(reinterpret_cast<char*>(&m),8); meta.read(reinterpret_cast<char*>(&n),8);
  auto pos=meta.tellg(); meta.seekg(0,std::ios::end); auto end=meta.tellg();
  int64_t expected=args->model==model_name::sup ? dict->nlabels() : dict->nwords();
  require(bool(meta) && m==expected && m>=0 && n==int64_t(dim) && pos>=0 && end>=pos &&
          uint64_t(end-pos)==uint64_t(m)*dim*4,"invalid native output matrix extent");
  meta.seekg(pos);
  auto output=std::make_shared<Matrix>(m,n);
  meta.read(reinterpret_cast<char*>(output->data()),uint64_t(m)*dim*4);
  require(bool(meta),"truncated native output matrix");
  auto backend=std::make_shared<NativeInput>(directory,rows,dim);
  require(backend->keyCount()==static_cast<size_t>(dict->nngrams()),"native index key count differs from dictionary");
  int32_t words=dict->nwords(), ngrams=dict->nngrams();
  dict->setNativeNgramLookup([backend,words,ngrams](const std::string& key) {return backend->lookup(key,words,ngrams);});
  auto empty=std::make_shared<Matrix>();
  auto model=std::make_shared<Model>(empty,output,args,0);
  model->quant_=false; model->setInputBackend(backend);
  if (args->loss==loss_name::hs) model->setTargetCounts(dict->getCounts(args->model==model_name::sup ? entry_type::label : entry_type::word));
  args_=args; dict_=dict; input_=empty; output_=output;
  qinput_.reset(); qoutput_.reset(); model_=model; input_backend_=backend;
  quant_=false; version=12;
}
}
