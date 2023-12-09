CC=g++
INCLUDE=$(shell pkg-config --cflags opencv4)			# 获得 opencv 相关头文件路径
LIBS=$(shell pkg-config --libs opencv4) -ljsoncpp		# 获得 opencv 扩展库
CFLAGS=-O3
DEFS=
SRC=$(sort $(wildcard src/*.cpp))
OBJ=$(SRC:.cpp=.o)
TARGET=eifenface.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) $(DEFS) $(OBJ) -o $@

$(OBJ): %.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) $(DEFS) -c $< -o $@

clean: cleanbin cleanimg cleancfg

cleanbin:
	rm -rf $(OBJ) $(TARGET)

cleanimg:
	find . -type f -name "*.jpg" -o -name "*.png" -not -path "./pictures/*" -delete
	rm -rf inner/

cleancfg:
	rm -rf eigenface.yml