CC=g++
INCLUDE=-I /usr/include/opencv4
LIB=-L /usr/lib
LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CFLAGS=-g
TARGET=canny

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CC) $(CFLAGS) $(LIB) $(LIBS) $(TARGET).o -o $(TARGET).out

$(TARGET).o: $(TARGET).cpp
	$(CC) $(INCLUDE) -c $(TARGET).cpp

clean:
	rm -rf *o $(TARGET).out