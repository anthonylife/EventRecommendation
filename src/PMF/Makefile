CC = g++
CFLAGS = -Wall -O3 -fopenmp -Wcpp 

all: train

train: main.cpp ../utils.hpp model.hpp
	$(CC) $(CFLAGS) -o run main.cpp 

clean:
	rm run
