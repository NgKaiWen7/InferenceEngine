CC = g++
CFLAGS = -Wall -g -std=c++17 $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4) \
       -L./onnxruntime/lib -lonnxruntime \
       -Wl,-rpath,'$$ORIGIN/onnxruntime/lib'

TARGET = main

$(TARGET): main.cpp
	$(CC) $(CFLAGS) main.cpp $(LIBS) -I./onnxruntime/include -o $(TARGET)

clean:
	rm -f $(TARGET)