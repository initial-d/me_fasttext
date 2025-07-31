#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x -march=native -fPIC
OBJS = args.o dictionary.o productquantizer.o matrix.o qmatrix.o vector.o model.o utils.o fasttext.o ftindex.o
INCLUDES = -I. -I /home/search/ulike/model/fastText/mylib/include
LIBS = -L /home/search/ulike/model/fastText/mylib/lib -ltrie -lpub
LIBNAME = libfasttext

opt: CXXFLAGS += -O1 -funroll-loops
opt: fasttext $(LIBNAME).a index cut_ngram print_ngrams

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc $(INCLUDES)

productquantizer.o: src/productquantizer.cc src/productquantizer.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/productquantizer.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

qmatrix.o: src/qmatrix.cc src/qmatrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/qmatrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc $(INCLUDES)

ftindex.o: src/ftindex.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/ftindex.cc $(INCLUDES)

fasttext: $(OBJS) src/fasttext.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o fasttext $(INCLUDES) $(LIBS)
	ar -x ./mylib/lib/libtrie.a
	ar -x ./mylib/lib/libpub.a
#$(LIBNAME).a: $(OBJS)
#	ar r $(LIBNAME).a $^
$(LIBNAME).a: $(OBJS)
	ar r $(LIBNAME).a *.o

index: $(OBJS) src/index.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/index.cc -o index $(INCLUDES) $(LIBS)

cut_ngram: $(OBJS) src/cut_ngram.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/cut_ngram.cc -o cut_ngram $(INCLUDES) $(LIBS)

print_ngrams: $(OBJS) src/print_ngrams.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/print_ngrams.cc -o print_ngrams $(INCLUDES) $(LIBS)
	ar -x ./mylib/lib/libtrie.a
	ar -x ./mylib/lib/libpub.a
#$(LIBNAME).a: $(OBJS)
#	ar r $(LIBNAME).a $^
$(LIBNAME).a: $(OBJS)
	ar r $(LIBNAME).a *.o

clean:
	rm -rf *.o fasttext $(LIBNAME).a
