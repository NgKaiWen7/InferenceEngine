MKLROOT ?= /opt/intel/oneapi/mkl/2026.1
ICOMPROOT ?= /opt/intel/oneapi/compiler/2026.1

CXXFLAGS += -std=c++23 -O3 -march=native -fopenmp -DNDEBUG

INCLUDES := -Iinclude -I"$(MKLROOT)/include"
LIBDIRS  := -L"$(MKLROOT)/lib/intel64" -Wl,-rpath,"$(MKLROOT)/lib/intel64" \
            -L"$(ICOMPROOT)/lib" -Wl,-rpath,"$(ICOMPROOT)/lib"

# Threaded MKL (NOT mkl_sequential — see prior gotcha)
MKL_LIBS := -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

SRCS := main.cpp \
        include/tokenizer/BGEtokenizer.cpp \
        include/embedding/embedding.cpp \
        include/utils/conversion.cpp \
        include/utils/immitrin.cpp \
        include/attention/self_attention.cpp

xlmr-blas: $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) $(LIBDIRS) \
		-licuuc -licui18n -lsentencepiece $(MKL_LIBS) -lpthread -lm -ldl \
		-o main
