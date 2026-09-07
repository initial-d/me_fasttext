#include "ftindex.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double micros_since(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start)
      .count();
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double pos = q * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, values.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::vector<std::string> read_queries(const char* path) {
  std::ifstream in(path);
  std::vector<std::string> queries;
  std::string line;

  while (std::getline(in, line)) {
    if (!line.empty()) {
      queries.push_back(line);
    }
  }
  return queries;
}

double l2_norm(const std::vector<real>& vec) {
  double sum = 0.0;
  for (real value : vec) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  return std::sqrt(sum);
}

void sentence_vector(FTIndex* index, const std::string& sentence, std::vector<real>& out) {
  std::fill(out.begin(), out.end(), 0.0f);

  std::istringstream stream(sentence);
  std::string token;
  std::vector<real> word_vec(out.size(), 0.0f);
  size_t count = 0;

  while (stream >> token) {
    getWordVector(index, token, word_vec.data());
    const double norm = l2_norm(word_vec);
    if (norm <= 0.0) {
      continue;
    }
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] += static_cast<real>(static_cast<double>(word_vec[i]) / norm);
    }
    ++count;
  }

  if (count > 1) {
    for (real& value : out) {
      value = static_cast<real>(value / static_cast<real>(count));
    }
  }
}

void print_usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " <model.z> <queries.txt> [warmup_rounds=1] [measure_rounds=3]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    print_usage(argv[0]);
    return 1;
  }

  const char* model_path = argv[1];
  const char* query_path = argv[2];
  const int warmup_rounds = std::max(0, argc >= 4 ? std::atoi(argv[3]) : 1);
  const int measure_rounds = std::max(1, argc >= 5 ? std::atoi(argv[4]) : 3);

  std::vector<std::string> queries = read_queries(query_path);
  if (queries.empty()) {
    std::cerr << "query file is empty: " << query_path << "\n";
    return 1;
  }

  void* mmap_addr = nullptr;
  size_t mmap_len = 0;
  const auto load_start = Clock::now();
  FTIndex* index = ftindex_mmap(model_path, &mmap_addr, &mmap_len);
  const auto load_end = Clock::now();
  if (!index) {
    std::cerr << "failed to mmap index: " << model_path << "\n";
    return 1;
  }

  const int32_t dim = ftindex_dim(index);
  if (dim <= 0) {
    std::cerr << "invalid index dimension: " << dim << "\n";
    ftindex_munmap(mmap_addr, mmap_len);
    ftindex_delete(index);
    return 1;
  }
  std::vector<real> vec(static_cast<size_t>(dim), 0.0f);
  std::vector<real> scratch(static_cast<size_t>(dim), 0.0f);

  size_t total_tokens = 0;
  size_t oov_tokens = 0;
  size_t oov_with_subwords = 0;
  for (const std::string& query : queries) {
    std::istringstream stream(query);
    std::string token;
    while (stream >> token) {
      ++total_tokens;
      if (getWordId(index, token) < 0) {
        ++oov_tokens;
        if (getWordVectorCnt(index, token, scratch.data()) > 0) {
          ++oov_with_subwords;
        }
      }
    }
  }

  volatile double checksum = 0.0;
  for (int round = 0; round < warmup_rounds; ++round) {
    for (const std::string& query : queries) {
      sentence_vector(index, query, vec);
      checksum += vec.empty() ? 0.0 : vec[0];
    }
  }

  std::vector<double> latencies;
  latencies.reserve(queries.size() * static_cast<size_t>(std::max(1, measure_rounds)));
  for (int round = 0; round < measure_rounds; ++round) {
    for (const std::string& query : queries) {
      const auto start = Clock::now();
      sentence_vector(index, query, vec);
      const auto end = Clock::now();
      checksum += vec.empty() ? 0.0 : vec[0];
      latencies.push_back(micros_since(start, end));
    }
  }

  const double mean =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  const double oov_rate =
      total_tokens == 0 ? 0.0 : static_cast<double>(oov_tokens) / total_tokens;
  const double oov_coverage = oov_tokens == 0
      ? 1.0
      : static_cast<double>(oov_with_subwords) / oov_tokens;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "| metric | value |\n";
  std::cout << "| --- | ---: |\n";
  std::cout << "| cold_mmap_load_ms | " << micros_since(load_start, load_end) / 1000.0
            << " |\n";
  std::cout << "| mmap_bytes | " << mmap_len << " |\n";
  std::cout << "| dim | " << dim << " |\n";
  std::cout << "| nwords | " << ftindex_nwords(index) << " |\n";
  std::cout << "| nngrams | " << ftindex_nngrams(index) << " |\n";
  std::cout << "| queries | " << queries.size() << " |\n";
  std::cout << "| total_tokens | " << total_tokens << " |\n";
  std::cout << "| oov_tokens | " << oov_tokens << " |\n";
  std::cout << "| oov_rate | " << oov_rate << " |\n";
  std::cout << "| oov_subword_coverage | " << oov_coverage << " |\n";
  std::cout << "| query_mean_us | " << mean << " |\n";
  std::cout << "| query_p50_us | " << percentile(latencies, 0.50) << " |\n";
  std::cout << "| query_p95_us | " << percentile(latencies, 0.95) << " |\n";
  std::cout << "| checksum | " << checksum << " |\n";

  ftindex_munmap(mmap_addr, mmap_len);
  ftindex_delete(index);
  return 0;
}
