// Real DATrie -> .z export -> DASTrie mmap regression. Deterministic vectors
// below are correctness controls, not trained embedding/compression evidence.
#include "fasttext.h"
#include "ftindex.h"
#include "row_merge.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>

using Row = std::vector<float>;
const std::vector<std::string> keys = {
    "ab", "abc", "abcd", "bc", "xbc", "xy", "xyz", "uv",
    "你好", "你好啊", "zz", "zzz", "aa", "aab"};

class FixtureDictionary : public fasttext::Dictionary {
 public:
  explicit FixtureDictionary(std::shared_ptr<fasttext::Args> args) : Dictionary(args) {
    worddat = datrie_new();
    ngramdat = datrie_new();
    words_.push_back({"known", 1, fasttext::entry_type::word, {0}});
    nwords_ = size_ = 1;
    nngrams_ = keys.size();
    datrie_insert_key(worddat, "known", 0);
    for (size_t i = 0; i < keys.size(); ++i) {
      // Deliberately assign non-lexical IDs: first traversed key is not row 0.
      datrie_insert_key(ngramdat, keys[i].c_str(), keys.size() - 1 - i);
    }
  }
};

class FixtureModel : public fasttext::FastText {
 public:
  std::map<std::string, Row> originals;
  Row word;

  FixtureModel(int dim, const std::string& output) {
    args_ = std::make_shared<fasttext::Args>();
    args_->dim = dim; args_->minn = 2; args_->maxn = 4; args_->output = output;
    dict_ = std::make_shared<FixtureDictionary>(args_);
    input_ = std::make_shared<fasttext::Matrix>(1 + keys.size(), dim);
    input_->zero();
    word.resize(dim);
    for (int j = 0; j < dim; ++j) word[j] = input_->at(0, j) = 0.125f + j / 1024.f;
    for (const std::string& key : keys) originals[key] = Row(dim, 0);
    for (const std::string& key : {"ab", "abc", "uv"}) originals[key][0] = 1;
    originals["abcd"][0] = -1;
    originals["bc"][1] = originals["xbc"][1] = 1;
    originals["xy"][2] = 1; originals["xyz"][2] = 2;
    originals["你好"][3] = originals["你好啊"][3] = 1;
    originals["aa"][6] = 1;
    const double angle = 7 * std::acos(-1.0) / 180;
    originals["aab"][6] = std::cos(angle); originals["aab"][7] = std::sin(angle);
    for (const std::string& key : keys) {
      Row& row = originals[key];
      // Tail data detects export copies that still stop at component 128.
      if (key != "zz" && key != "zzz") row[dim-1] = key == "abcd" ? -.25f : .25f;
      if (key == "xyz") row[dim-1] = .5f;
      size_t id;
      assert(datrie_search_exact_match(dict_->ngramdat, key.c_str(), &id) != -1);
      std::memcpy(input_->data() + (id + 1) * dim, row.data(), dim * sizeof(float));
    }
  }
};

void verify(const std::string& path, const FixtureModel& model, int dim) {
  void* address = nullptr;
  size_t bytes = 0;
  FTIndex* index = ftindex_mmap(path.c_str(), &address, &bytes);
  assert(index && address && bytes);
  assert(ftindex_dim(index) == dim && ftindex_nwords(index) == 1);
  assert(ftindex_nngrams(index) > 0 && ftindex_nngrams(index) < int(keys.size()));
  Row recovered(dim);
  for (const std::string& key : keys) {
    assert(getNgramId(index, key) >= 0);
    getNgramVector(index, key, recovered.data());
    assert(fasttext::joint_row_similarity(recovered.data(), model.originals.at(key).data(), dim));
    if (key != "aab") {
      assert(std::memcmp(recovered.data(), model.originals.at(key).data(), dim*sizeof(float)) == 0);
    }
    bool has_representative = false;
    for (const auto& entry : model.originals) {
      if (std::memcmp(recovered.data(), entry.second.data(), dim*sizeof(float)) == 0 &&
          fasttext::structurally_related(key, entry.first)) has_representative = true;
    }
    assert(has_representative);
  }
  assert(getNgramId(index, "ab") == getNgramId(index, "abc"));
  assert(getNgramId(index, "bc") == getNgramId(index, "xbc"));
  assert(getNgramId(index, "你好") == getNgramId(index, "你好啊"));
  assert(getNgramId(index, "aa") == getNgramId(index, "aab"));
  assert(getNgramId(index, "ab") != getNgramId(index, "abcd"));
  assert(getNgramId(index, "xy") != getNgramId(index, "xyz"));
  assert(getNgramId(index, "ab") != getNgramId(index, "uv"));
  getWordVector(index, "known", recovered.data());
  assert(std::memcmp(recovered.data(), model.word.data(), dim*sizeof(float)) == 0);
  // OOV serving must average exactly the subwords available in the mapped trie.
  std::vector<uint32_t> ids;
  std::vector<std::string> subwords;
  computeSubwords(index, "<aabx>", ids, subwords);
  assert(!ids.empty());
  Row expected(dim, 0), row(dim);
  for (const std::string& key : subwords) {
    getNgramVector(index, key, row.data());
    for (int j = 0; j < dim; ++j) expected[j] += row[j];
  }
  for (float& value : expected) value /= ids.size();
  getWordVector(index, "aabx", recovered.data());
  assert(std::memcmp(recovered.data(), expected.data(), dim*sizeof(float)) == 0);
  assert(getNgramId(index, "missing") == -1);
  getNgramVector(index, "missing", recovered.data());
  for (float value : recovered) assert(value == 0);
  // Delete both trie views while their mapped backing bytes are still valid.
  ftindex_delete(index);
  assert(ftindex_munmap(address, bytes) == 0);
}

int main(int argc, char** argv) {
  assert(argc == 2);
  for (int dim : {64, 128, 256}) {
    const std::string base = std::string(argv[1]) + "/fixture-" + std::to_string(dim);
    FixtureModel model(dim, base);
    model.saveIndex();
    verify(base + ".z", model, dim);
    model.saveIndex(base + "-explicit.z");
    verify(base + "-explicit.z", model, dim);
    std::cout << "roundtrip dim=" << dim << " overloads=2 ngrams=" << keys.size()
              << " word/OOV/row checks passed" << std::endl;
  }
}
