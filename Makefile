CXX = g++
CXXFLAGS = -std=c++23 -O2 -Iinclude

tokenizer: clean tokenizer.cpp include/tokenizer/UnicodeEncoder.cpp include/tokenizer/BPE.cpp
	$(CXX) $(CXXFLAGS) tokenizer.cpp include/tokenizer/UnicodeEncoder.cpp include/tokenizer/BPE.cpp -licuuc -licui18n -o tokenizer

main: main.cpp \
include/utils/conversion.cpp \
include/embedding/embedding.cpp \
include/tokenizer/BPE.cpp \
include/tokenizer/UnicodeEncoder.cpp \
include/attention/casual_attention.cpp
	$(CXX) $(CXXFLAGS) \
	    main.cpp \
	    include/utils/conversion.cpp \
	    include/embedding/embedding.cpp \
		include/tokenizer/BPE.cpp \
		include/tokenizer/UnicodeEncoder.cpp \
		include/attention/casual_attention.cpp \
	    -licuuc -licui18n \
	    -o main
		
xlmr: main.cpp \
	include/tokenizer/BGEtokenizer.cpp \
	include/utils/conversion.cpp \
	include/embedding/embedding.cpp \
	include/attention/self_attention.cpp
	$(CXX) $(CXXFLAGS) \
	    main.cpp \
		include/tokenizer/BGEtokenizer.cpp \
		include/embedding/embedding.cpp \
	    include/utils/conversion.cpp \
		include/attention/self_attention.cpp \
	    -licuuc -licui18n -lsentencepiece \
	    -o main

safetensor_loader:
	$(CXX) $(CXXFLAGS) safe_tensor.cpp -o safe_tensor

clean:
	rm -f tokenizer
	rm -f safe_tensor