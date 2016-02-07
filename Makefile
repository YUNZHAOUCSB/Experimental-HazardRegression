CXX = g++
DEPS_PATH = $(shell pwd)/deps

INCPATH = -I./src -I./include -I./dmlc-core/include -I./dmlc-core/src -I$(DEPS_PATH)/include
CFLAGS = -std=c++11 -fopenmp -fPIC -O3 -ggdb -Wall -finline-functions $(INCPATH) -DDMLC_LOG_FATAL_THROW=0

OBJS = $(addprefix build/,  \
updater.o sgd/sgd_updater.o \
learner.o sgd/sgd_learner.o \
data/batch_iter.o )

DMLC_DEPS = dmlc-core/libdmlc.a

all: build/hazard

clean:
	rm -rf build
	make -C dmlc-core clean

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++0x -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

build/libhazard.a: $(OBJS)
	ar crv $@ $(filter %.o, $?)

build/hazard: build/main.o build/libhazard.a $(DMLC_DEPS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

dmlc-core/libdmlc.a:
	$(MAKE) -C dmlc-core libdmlc.a DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)


-include build/*.d
-include build/*/*.d
