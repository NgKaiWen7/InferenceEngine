CXX = g++
CXXFLAGS = -std=c++17 -O2 -Iinclude

tokenizer: clean tokenizer.cpp include/tokenizer/UnicodeEncoder.cpp include/tokenizer/BPE.cpp
	$(CXX) $(CXXFLAGS) tokenizer.cpp include/tokenizer/UnicodeEncoder.cpp include/tokenizer/BPE.cpp -licuuc -licui18n -o tokenizer

safetensor_loader:
	$(CXX) $(CXXFLAGS) safe_tensor.cpp -o safe_tensor

clean:
	rm -f tokenizer
	rm -f safe_tensor