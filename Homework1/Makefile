CC=g++
INCLUDE=-I /usr/include/opencv4
LIB=-L /usr/lib
LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CFLAGS=
DEFS=-D USE_WEAK_EDGE
SRC=$(sort $(wildcard src/*.cpp))
OBJ=$(SRC:.cpp=.o)
TARGET=canny.out

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDE) $(LIB) $(LIBS) $(DEFS) $(OBJ) -o $@

$(OBJ):
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIB) $(LIBS) $(DEFS) -c $< -o $@

clean:
	rm -rf $(OBJ) $(TARGET)

cleanimg:
	rm -rf inner/*.png inner/ *.png *.jpg