CC=g++
INCLUDE=$(shell pkg-config --cflags opencv4)			# 获得 opencv 相关头文件路径
LIBS=$(shell pkg-config --libs opencv4) -ljsoncpp		# 获得 opencv 扩展库
CFLAGS=-O3
DEFS=
SRC=$(sort $(wildcard src/*.cpp))
OBJ=$(SRC:.cpp=.o)
TARGET=eifenface.out
TESTSRC=$(sort $(wildcard src/test/*.cpp))
TESTOBJ=$(TESTSRC:.cpp=.o)
TESTTARGET=test.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) $(DEFS) $(OBJ) -o $@

$(OBJ): %.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) $(DEFS) -c $< -o $@

test: $(TESTTARGET)

$(TESTTARGET): $(TESTOBJ) src/utils.o src/face.o
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) $(DEFS) $(TESTOBJ) src/utils.o src/face.o -o $@

$(TESTOBJ): %.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) $(DEFS) -c $< -o $@

clean: cleanbin cleanimg cleancfg

cleanbin:
	rm -rf $(OBJ) $(TARGET) $(TESTOBJ) $(TESTTARGET)

cleanimg:
	rm -rf inner/ *.jpg *.png *.bmp *.svg

cleancfg:
	find . -type f -name "*.yml" -delete