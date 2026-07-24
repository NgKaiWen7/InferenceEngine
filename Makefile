CXX = g++
CXXFLAGS = -std=c++17 -O2 -Iinclude

json_loader:
	$(CXX) $(CXXFLAGS) main.cpp -o json_loader

clean:
	rm -f json_loader